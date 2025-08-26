#!/usr/bin/env python3
"""
Enhanced Parkers Scraper that:
1. Reads Series + URL.xlsx file
2. Uses resolver_clean.py to resolve specs URLs to review URLs
3. Scrapes review content and formats as JSON like parkers_article.json
4. Populates column C in Excel with JSON data
5. Processes URLs in batches with concurrency control
"""

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import traceback

import httpx
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from selectolax.parser import HTMLParser
import openpyxl
from openpyxl import load_workbook

# Import our clean resolver
from resolver_clean import resolve_via_specs_tabs_playwright

# ==============================
# Config & Helpers
# ==============================

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/114.0.0.0 Safari/537.36")
}

COOKIE_BUTTON_SELECTORS = [
    "button#onetrust-accept-btn-handler",
    "button[aria-label='Accept all']",
    "button:has-text('Accept all cookies')",
    "button:has-text('Accept All Cookies')",
    "button:has-text('Accept')",
    "button:has-text('Agree')",
    "button[id*='accept']",
    "button[class*='accept']",
]

@dataclass
class ScrapeConfig:
    headless: bool = True
    timeout: int = 30000
    save_debug_html: bool = False
    retry_count: int = 3

def normalize_ws(s: Optional[str]) -> str:
    """Normalize whitespace in strings."""
    return re.sub(r"\s+", " ", (s or "")).strip()

def get_content_after_element(element, tree) -> Optional[str]:
    """Get text content that appears after a heading element."""
    try:
        # Get all siblings after this element
        parent = element.parent
        if not parent:
            return None
        
        # Find this element's position
        siblings = parent.child
        if not siblings:
            return None
        
        # Get text from subsequent paragraphs/divs
        content_parts = []
        found_element = False
        
        for sibling in siblings:
            if sibling == element:
                found_element = True
                continue
            
            if found_element:
                if hasattr(sibling, 'tag') and sibling.tag in ['p', 'div', 'section']:
                    text = normalize_ws(sibling.text())
                    if text and len(text) > 10:
                        content_parts.append(text)
                        if len(content_parts) >= 3:  # Limit to 3 paragraphs
                            break
        
        return " ".join(content_parts) if content_parts else None
    except:
        return None

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove common unwanted elements
    text = re.sub(r'© \d{4}-\d{4}.*?$', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'Bauer Media Group.*?$', '', text, flags=re.MULTILINE | re.DOTALL)
    return text.strip()

def extract_key_info_from_sections(sections: Dict[str, str], html_content: str) -> Dict[str, str]:
    """Extract key information from sections into structured format."""
    key_info = {}
    
    # Extract price information
    if "At a glance" in sections:
        glance_text = sections["At a glance"]
        # Extract new price
        new_price_match = re.search(r'Price new £([\d,]+) - £([\d,]+)', glance_text)
        if new_price_match:
            key_info["new_price_min"] = new_price_match.group(1).replace(',', '')
            key_info["new_price_max"] = new_price_match.group(2).replace(',', '')
        
        # Extract used price
        used_price_match = re.search(r'Used prices £([\d,]+) - £([\d,]+)', glance_text)
        if used_price_match:
            key_info["used_price_min"] = used_price_match.group(1).replace(',', '')
            key_info["used_price_max"] = used_price_match.group(2).replace(',', '')
        
        # Extract road tax
        tax_match = re.search(r'Road tax cost £([\d,]+) - £([\d,]+)', glance_text)
        if tax_match:
            key_info["road_tax_min"] = tax_match.group(1).replace(',', '')
            key_info["road_tax_max"] = tax_match.group(2).replace(',', '')
        
        # Extract insurance group
        insurance_match = re.search(r'Insurance group ([\d-]+)', glance_text)
        if insurance_match:
            key_info["insurance_group"] = insurance_match.group(1)
        
        # Extract fuel economy
        fuel_match = re.search(r'Fuel economy ([^R]+) Range ([\d,]+) - ([\d,]+) miles', glance_text)
        if fuel_match:
            key_info["fuel_economy"] = fuel_match.group(1).strip()
            key_info["range_min"] = fuel_match.group(2).replace(',', '')
            key_info["range_max"] = fuel_match.group(3).replace(',', '')
        
        # Extract doors
        doors_match = re.search(r'Number of doors ([\d]+)', glance_text)
        if doors_match:
            key_info["doors"] = doors_match.group(1)
        
        # Extract fuel types
        fuel_types_match = re.search(r'Available fuel types ([^V]+)', glance_text)
        if fuel_types_match:
            key_info["fuel_types"] = fuel_types_match.group(1).strip()
    
    # Extract pros and cons
    if "Pros & cons" in sections:
        pros_cons = sections["Pros & cons"]
        # Better regex patterns for pros and cons
        pros_match = re.search(r'PROS\s*(.*?)(?=CONS|$)', pros_cons, re.DOTALL | re.IGNORECASE)
        cons_match = re.search(r'CONS\s*(.*?)(?=PROS|$)', pros_cons, re.DOTALL | re.IGNORECASE)
        
        if pros_match:
            pros_text = clean_text(pros_match.group(1))
            # Clean up the pros text - remove extra whitespace and normalize
            pros_text = re.sub(r'\s+', ' ', pros_text).strip()
            if pros_text and pros_text != '&':
                key_info["pros"] = pros_text
        if cons_match:
            cons_text = clean_text(cons_match.group(1))
            # Clean up the cons text - remove extra whitespace and normalize
            cons_text = re.sub(r'\s+', ' ', cons_text).strip()
            if cons_text and cons_text != '&':
                key_info["cons"] = cons_text
    
    # Also try to extract pros/cons from individual sections if available
    if "Pros" in sections and not key_info.get("pros"):
        pros_text = sections["Pros"]
        if pros_text and pros_text != '&':
            key_info["pros"] = clean_text(pros_text)
    
    if "Cons" in sections and not key_info.get("cons"):
        cons_text = sections["Cons"]
        if cons_text and cons_text != '&':
            key_info["cons"] = clean_text(cons_text)
    
    # If still no pros/cons found, try to extract from the HTML content directly
    if not key_info.get("pros") or not key_info.get("cons"):
        # Look for common pros/cons patterns in the HTML
        html_lower = html_content.lower()
        
        # Try to find pros and cons using various text patterns
        pros_patterns = [
            r'pros?[:\s]+(.*?)(?=cons?|verdict|overview|$)', 
            r'advantages?[:\s]+(.*?)(?=disadvantages?|verdict|overview|$)',
            r'good points?[:\s]+(.*?)(?=bad points?|verdict|overview|$)',
            r'strengths?[:\s]+(.*?)(?=weaknesses?|verdict|overview|$)',
            r'benefits?[:\s]+(.*?)(?=drawbacks|verdict|overview|$)'
        ]
        cons_patterns = [
            r'cons?[:\s]+(.*?)(?=pros?|verdict|overview|$)', 
            r'disadvantages?[:\s]+(.*?)(?=advantages?|verdict|overview|$)',
            r'bad points?[:\s]+(.*?)(?=good points?|verdict|overview|$)',
            r'weaknesses?[:\s]+(.*?)(?=strengths?|verdict|overview|$)',
            r'drawbacks?[:\s]+(.*?)(?=benefits|verdict|overview|$)'
        ]
        
        # Try to extract pros
        if not key_info.get("pros"):
            for pattern in pros_patterns:
                pros_match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
                if pros_match:
                    pros_text = clean_text(pros_match.group(1))
                    if pros_text and len(pros_text) > 5 and pros_text != '&':
                        # Clean up the text
                        pros_text = re.sub(r'\s+', ' ', pros_text).strip()
                        # Limit length to avoid very long text
                        if len(pros_text) < 1000:
                            key_info["pros"] = pros_text
                            break
        
        # Try to extract cons
        if not key_info.get("cons"):
            for pattern in cons_patterns:
                cons_match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
                if cons_match:
                    cons_text = clean_text(cons_match.group(1))
                    if cons_text and len(cons_text) > 5 and cons_text != '&':
                        # Clean up the text
                        cons_text = re.sub(r'\s+', ' ', cons_text).strip()
                        # Limit length to avoid very long text
                        if len(cons_text) < 1000:
                            key_info["cons"] = cons_text
                            break
    
    # Extract author and date
    if "About the author" in sections:
        author_text = sections["About the author"]
        author_match = re.search(r'Written by ([^P]+) Published: ([^W]+)', author_text)
        if author_match:
            key_info["author"] = author_match.group(1).strip()
            key_info["publish_date"] = author_match.group(2).strip()
    
    # Also try to extract from the HTML directly if not found in sections
    if not key_info.get("author"):
        # Try to find in the page content
        author_match = re.search(r'Written by ([^,\n]+)', html_content)
        if author_match:
            key_info["author"] = author_match.group(1).strip()
    
    # Extract rivals - prioritize the new structured format with URLs
    if "Rivals" in sections and not key_info.get("rivals"):
        rivals_text = sections["Rivals"]
        if rivals_text:
            # The new format includes URLs, so we can use it directly
            key_info["rivals"] = clean_text(rivals_text)
    
    # Also try to extract rivals from the Rivals section if available
    if "Rivals" in sections and not key_info.get("rivals"):
        rivals_text = sections["Rivals"]
        if rivals_text:
            key_info["rivals"] = clean_text(rivals_text)
    
    # Extract rivals with URLs from HTML content
    if not key_info.get("rivals"):
        # Try to find rivals in the HTML with potential URLs
        rivals_pattern = r'<a[^>]*href="([^"]*)"[^>]*>([^<]+)</a>'
        rival_matches = re.findall(rivals_pattern, html_content)
        if rival_matches:
            rivals_with_urls = []
            for url, name in rival_matches:
                if 'rival' in url.lower() or 'competitor' in url.lower():
                    rivals_with_urls.append(f"{name.strip()}: {url}")
            if rivals_with_urls:
                key_info["rivals"] = " | ".join(rivals_with_urls)
    
    # Fallback: Extract rivals from the old format if nothing else worked
    if "Abarth 500 Hatchback rivals" in sections and not key_info.get("rivals"):
        rivals_text = sections["Abarth 500 Hatchback rivals"]
        # Extract rival cars and ratings
        rivals = []
        rival_matches = re.findall(r'([A-Za-z\s]+)\s+([\d.]+)\s+out of\s+5', rivals_text)
        for rival_name, rating in rival_matches:
            rivals.append(f"{rival_name.strip()}: {rating}/5")
        if rivals:
            key_info["rivals"] = " | ".join(rivals)
    
    return key_info

def extract_sections_from_html(html_content: str) -> Dict[str, Any]:
    """Extract structured data from HTML content using selectolax."""
    parser = HTMLParser(html_content)
    
    # Extract title
    title = ""
    title_elem = parser.css_first('h1')
    if title_elem:
        title = clean_text(title_elem.text())
    
    # Extract main article text
    article_text = ""
    article_elem = parser.css_first('main, .article-content, .review-content')
    if article_elem:
        # Get text content, excluding navigation and ads
        text_parts = []
        for elem in article_elem.css('p, h2, h3, h4, h5, h6'):
            if elem.text().strip():
                text_parts.append(elem.text().strip())
        article_text = ' '.join(text_parts)
    
    # Extract sections
    sections = {}
    
    # Look for specific section headers and content
    section_headers = [
        'Overview', 'Practicality & safety', 'Interior, tech & comfort', 
        'Engines & handling', 'Ownership cost', 'Verdict', 'Pros & cons',
        'At a glance', 'About the author'
    ]
    
    for header in section_headers:
        # Find elements containing the header text
        header_elements = parser.css('h1, h2, h3, h4, h5, h6')
        for elem in header_elements:
            if header.lower() in elem.text().lower():
                # Get content after this header
                content = get_content_after_element(parser, elem)
                if content:
                    sections[header] = clean_text(content)
                break
    
    # Special handling for "At a glance" section
    glance_elem = parser.css_first('.at-a-glance, .specs-overview, [class*="glance"]')
    if glance_elem:
        sections["At a glance"] = clean_text(glance_elem.text())
    
    # Special handling for pros and cons
    pros_cons_elem = parser.css_first('.review-details-introduction__pros-cons, .pros-cons, .pros-and-cons, [class*="pros"]')
    if pros_cons_elem:
        # Extract pros and cons separately for better structure
        pros_elem = pros_cons_elem.css_first('.review-details-introduction__pros-cons__header--pros, [class*="pros"]')
        cons_elem = pros_cons_elem.css_first('.review-details-introduction__pros-cons__header--cons, [class*="cons"]')
        
        if pros_elem:
            # Get the ul element that follows the pros header
            pros_ul = pros_elem.parent.css_first('ul')
            if pros_ul:
                pros_items = [li.text().strip() for li in pros_ul.css('li')]
                if pros_items:
                    sections["Pros"] = ' | '.join(pros_items)
        
        if cons_elem:
            # Get the ul element that follows the cons header
            cons_ul = cons_elem.parent.css_first('ul')
            if cons_ul:
                cons_items = [li.text().strip() for li in cons_ul.css('li')]
                if cons_items:
                    sections["Cons"] = ' | '.join(cons_items)
        
        # Also store the full pros & cons section
        sections["Pros & cons"] = clean_text(pros_cons_elem.text())
    
    # More robust pros/cons extraction - look for various patterns
    if "Pros" not in sections or "Cons" not in sections:
        # Look for pros and cons in the entire HTML content
        html_lower = html_content.lower()
        
        # Try to find pros and cons sections using various selectors
        pros_selectors = [
            '.pros', '[class*="pros"]', '.advantages', '.benefits',
            '.good-points', '.strengths', '.positives'
        ]
        cons_selectors = [
            '.cons', '[class*="cons"]', '.disadvantages', '.drawbacks',
            '.bad-points', '.weaknesses', '.negatives'
        ]
        
        # Look for pros
        if "Pros" not in sections:
            for selector in pros_selectors:
                pros_elem = parser.css_first(selector)
                if pros_elem:
                    pros_text = clean_text(pros_elem.text())
                    if pros_text and len(pros_text) > 5 and pros_text != '&':
                        sections["Pros"] = pros_text
                        break
        
        # Look for cons
        if "Cons" not in sections:
            for selector in cons_selectors:
                cons_elem = parser.css_first(selector)
                if cons_elem:
                    cons_text = clean_text(cons_elem.text())
                    if cons_text and len(cons_text) > 5 and cons_text != '&':
                        sections["Cons"] = cons_text
                        break
        
        # If still not found, try to extract from text content using regex
        if "Pros" not in sections or "Cons" not in sections:
            # Look for text patterns like "Pros:" or "Cons:" in the content
            pros_patterns = [
                r'pros?[:\s]+(.*?)(?=cons?|$)', 
                r'advantages?[:\s]+(.*?)(?=disadvantages?|$)',
                r'good points?[:\s]+(.*?)(?=bad points?|$)',
                r'strengths?[:\s]+(.*?)(?=weaknesses?|$)'
            ]
            cons_patterns = [
                r'cons?[:\s]+(.*?)(?=pros?|$)', 
                r'disadvantages?[:\s]+(.*?)(?=advantages?|$)',
                r'bad points?[:\s]+(.*?)(?=good points?|$)',
                r'weaknesses?[:\s]+(.*?)(?=strengths?|$)'
            ]
            
            # Try to extract pros
            if "Pros" not in sections:
                for pattern in pros_patterns:
                    pros_match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
                    if pros_match:
                        pros_text = clean_text(pros_match.group(1))
                        if pros_text and len(pros_text) > 5 and pros_text != '&':
                            sections["Pros"] = pros_text
                            break
            
            # Try to extract cons
            if "Cons" not in sections:
                for pattern in cons_patterns:
                    cons_match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
                    if cons_match:
                        cons_text = clean_text(cons_match.group(1))
                        if cons_text and len(cons_text) > 5 and cons_text != '&':
                            sections["Cons"] = cons_text
                            break
    
    # Alternative pros/cons extraction if the above didn't work
    if "Pros" not in sections or "Cons" not in sections:
        # Look for pros and cons in the text content
        pros_cons_text = pros_cons_elem.text() if pros_cons_elem else ""
        if pros_cons_text:
            # Try to extract pros and cons using regex
            pros_match = re.search(r'PROS\s*(.*?)(?=CONS|$)', pros_cons_text, re.DOTALL | re.IGNORECASE)
            cons_match = re.search(r'CONS\s*(.*?)(?=PROS|$)', pros_cons_text, re.DOTALL | re.IGNORECASE)
            
            if pros_match and "Pros" not in sections:
                pros_text = clean_text(pros_match.group(1))
                if pros_text and pros_text != '&':
                    sections["Pros"] = pros_text
            
            if cons_match and "Cons" not in sections:
                cons_text = clean_text(cons_match.group(1))
                if cons_text and cons_text != '&':
                    sections["Cons"] = cons_text
    
    # Special handling for rivals
    rivals_elem = parser.css_first('.review-details-introduction__rivals, .rivals, .competitors, [class*="rival"]')
    if rivals_elem:
        # Extract rivals with ratings and URLs
        rival_cards = rivals_elem.css('.rival-review-card')
        rivals_list = []
        
        for card in rival_cards:
            # Get rival name
            name_elem = card.css_first('.rival-review-card__content__text__model')
            if name_elem:
                rival_name = name_elem.text().strip()
                
                # Get the URL from the card's href attribute
                rival_url = card.attributes.get('href', '')
                if rival_url:
                    # Make URL absolute if it's relative
                    if rival_url.startswith('/'):
                        rival_url = f"https://www.parkers.co.uk{rival_url}"
                    
                    # Get rating if available
                    rating_elem = card.css_first('.star-rating__text')
                    if rating_elem:
                        rating = rating_elem.text().strip()
                        rivals_list.append(f"{rival_name}: {rating}/5 | {rival_url}")
                    else:
                        rivals_list.append(f"{rival_name} | {rival_url}")
                else:
                    # Fallback: just name and rating if no URL
                    rating_elem = card.css_first('.star-rating__text')
                    if rating_elem:
                        rating = rating_elem.text().strip()
                        rivals_list.append(f"{rival_name}: {rating}/5")
                    else:
                        rivals_list.append(rival_name)
        
        if rivals_list:
            sections["Rivals"] = ' | '.join(rivals_list)
        
        # Also store the full rivals section
        sections["Abarth 500 Hatchback rivals"] = clean_text(rivals_elem.text())
    
    # Create the main structure
    result = {
        "title": title,
        "article_text": clean_text(article_text),
        "sections": sections
    }
    
    # Extract key structured information
    key_info = extract_key_info_from_sections(sections, html_content)
    result["key_info"] = key_info
    
    return result

def get_content_after_element(parser: HTMLParser, element, max_elements: int = 5) -> str:
    """Extract text content from elements following a given element."""
    content_parts = []
    current = element.next
    
    count = 0
    while current and count < max_elements:
        if hasattr(current, 'text') and current.text().strip():
            content_parts.append(current.text().strip())
            count += 1
        current = current.next
    
    return ' '.join(content_parts)

async def scrape_review_page(url: str, config: ScrapeConfig) -> Dict[str, Any]:
    """
    Scrape a Parkers review page and return structured JSON data.
    """
    print(f"[Scraper] Starting scrape for: {url}")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=config.headless)
            ctx = await browser.new_context(user_agent=HEADERS["User-Agent"])
            page = await ctx.new_page()
            
            # Navigate to the page
            await page.goto(url, timeout=config.timeout * 1000, wait_until="domcontentloaded")
            
            # Handle cookies
            for selector in COOKIE_BUTTON_SELECTORS:
                try:
                    btn = await page.wait_for_selector(selector, timeout=2000)
                    if btn:
                        await btn.click()
                        print(f"[Scraper] Accepted cookies via: {selector}")
                        break
                except Exception:
                    continue
            
            # Wait for main content to load
            try:
                await page.wait_for_selector("main, article, .content", timeout=10000)
            except Exception:
                print("[Scraper] No main content selector found, proceeding anyway")
            
            # Get the page HTML
            html_content = await page.content()
            
            if config.save_debug_html:
                debug_path = f"debug_{int(time.time())}.html"
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"[Scraper] Saved debug HTML to: {debug_path}")
            
            await browser.close()
            
            # Extract structured content
            return extract_sections_from_html(html_content)
            
    except Exception as e:
        print(f"[Scraper] Error scraping {url}: {e}")
        return {
            "title": "Scraping failed",
            "article_text": f"Error: {e}",
            "article_markdown": f"Error: {e}",
            "sections": {"full_text": f"Error scraping {url}: {e}"},
            "url": url
        }

async def process_single_url(url: str, config: ScrapeConfig) -> Dict[str, Any]:
    """
    Process a single URL: resolve it if needed, then scrape content.
    Returns JSON data suitable for Excel column C.
    """
    print(f"\n[Processor] Processing URL: {url}")
    
    try:
        # Step 1: Check if this is a direct review URL that might not exist
        # If it ends with /review/, try to convert it to /specs/ first
        if url.endswith("/review/") or url.endswith("/review"):
            print(f"[Processor] Detected direct review URL, converting to specs URL first...")
            specs_url = url.replace("/review/", "/specs/").replace("/review", "/specs")
            print(f"[Processor] Will try specs URL: {specs_url}")
            
            # Use our resolver to get the working review URL
            resolved_url = await resolve_via_specs_tabs_playwright(specs_url)
            if resolved_url and resolved_url != specs_url:
                print(f"[Processor] Resolved specs to review: {resolved_url}")
                url = resolved_url
            else:
                print(f"[Processor] Resolution failed, trying original URL")
        elif url.endswith("/specs") or "/specs/" in url:
            print(f"[Processor] Detected specs URL, resolving to review URL...")
            resolved_url = await resolve_via_specs_tabs_playwright(url)
            if resolved_url and resolved_url != url:
                print(f"[Processor] Resolved to: {resolved_url}")
                url = resolved_url
            else:
                print(f"[Processor] Resolution returned same URL or None, using original")
        
        # Step 2: Scrape the review content
        result = await scrape_review_page(url, config)
        result["processing_status"] = "success"
        return result
        
    except Exception as e:
        print(f"[Processor] Error processing {url}: {e}")
        traceback.print_exc()
        return {
            "title": "Processing failed",
            "article_text": f"Error: {e}",
            "article_markdown": f"Error: {e}",
            "sections": {"full_text": f"Error processing {url}: {e}"},
            "url": url,
            "processing_status": "error",
            "error": str(e)
        }

async def process_url(url: str, config: ScrapeConfig) -> Dict[str, Any]:
    """Process a single URL to extract car review data."""
    try:
        # Check if URL is already a review URL
        if '/review' in url:
            # Convert to specs URL first
            specs_url = url.replace('/review', '/specs')
            print(f"Converting review URL to specs: {specs_url}")
            
            # Resolve to correct review URL
            resolved_url = await resolve_via_specs_tabs_playwright(specs_url)
            if resolved_url and resolved_url != specs_url:
                url = resolved_url
                print(f"Resolved to: {url}")
            else:
                print(f"Could not resolve review URL, using original: {url}")
        else:
            # Already a specs URL, resolve to review
            resolved_url = await resolve_via_specs_tabs_playwright(url)
            if resolved_url:
                url = resolved_url
                print(f"Resolved to: {url}")
            else:
                print(f"Could not resolve URL: {url}")
                return {"error": "Could not resolve URL", "url": url, "processing_status": "failed"}
        
        # Now scrape the resolved review URL with fallback to specs
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=config.headless)
            try:
                # Try review page first with extended timeout
                page = await browser.new_page()
                try:
                    print(f"Attempting to load review page: {url}")
                    await page.goto(url, wait_until='domcontentloaded', timeout=60000)  # 60 second timeout
                    
                    # Wait for content to load
                    await page.wait_for_timeout(3000)
                    
                    # Get the HTML content
                    html_content = await page.content()
                    
                    if config.save_debug_html:
                        debug_file = f"debug_review_{url.split('/')[-1]}.html"
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        print(f"Saved review page HTML to: {debug_file}")
                    
                    # Extract data from review page
                    extracted_data = extract_sections_from_html(html_content)
                    extracted_data["url"] = url
                    extracted_data["processing_status"] = "success"
                    extracted_data["source"] = "review_page"
                    
                    print(f"✅ Successfully scraped review page: {extracted_data.get('title', 'N/A')}")
                    return extracted_data
                    
                except Exception as review_error:
                    print(f"⚠️  Review page failed ({str(review_error)[:100]}...), falling back to specs page")
                    
                    # Fallback: Extract content from specs page instead
                    specs_url = url.replace('/review', '/specs').replace('/used-review', '/specs')
                    print(f"Fallback: Scraping specs page: {specs_url}")
                    
                    try:
                        await page.goto(specs_url, wait_until='domcontentloaded', timeout=30000)
                        await page.wait_for_timeout(2000)
                        
                        html_content = await page.content()
                        
                        if config.save_debug_html:
                            debug_file = f"debug_specs_fallback_{specs_url.split('/')[-1]}.html"
                            with open(debug_file, 'w', encoding='utf-8') as f:
                                f.write(html_content)
                            print(f"Saved specs fallback HTML to: {debug_file}")
                        
                        # Extract data from specs page
                        extracted_data = extract_sections_from_html(html_content)
                        extracted_data["url"] = specs_url
                        extracted_data["processing_status"] = "success"
                        extracted_data["source"] = "specs_page_fallback"
                        extracted_data["fallback_reason"] = str(review_error)[:200]
                        
                        print(f"✅ Successfully scraped specs page (fallback): {extracted_data.get('title', 'N/A')}")
                        return extracted_data
                        
                    except Exception as specs_error:
                        print(f"❌ Both review and specs pages failed")
                        return {
                            "error": f"Review page failed: {str(review_error)[:100]}... Specs fallback failed: {str(specs_error)[:100]}...",
                            "url": url,
                            "processing_status": "failed"
                        }
                
            finally:
                await browser.close()
                
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return {
            "error": str(e),
            "url": url,
            "processing_status": "failed"
        }

async def process_excel_file(excel_path: str, config: ScrapeConfig, max_rows: Optional[int] = None) -> None:
    """Process URLs from Excel file and update with scraped data."""
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Ensure we have the required columns
    if 'Model URL' not in df.columns:
        print("Error: 'Model URL' column not found in Excel file")
        return
    
    # Limit rows if specified
    if max_rows:
        df = df.head(max_rows)
        print(f"Processing limited to {max_rows} rows")
    
    # Add new columns for structured data
    new_columns = [
        'Title', 'Author', 'Publish_Date', 'New_Price_Min', 'New_Price_Max',
        'Used_Price_Min', 'Used_Price_Max', 'Road_Tax_Min', 'Road_Tax_Max',
        'Insurance_Group', 'Fuel_Economy', 'Range_Min', 'Range_Max', 'Doors',
        'Fuel_Types', 'Pros', 'Cons', 'Rivals', 'Article_Text', 'Full_JSON',
        'Processing_Status', 'Resolved_URL'
    ]
    
    for col in new_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Process URLs
    total_urls = len(df)
    successful = 0
    failed = 0
    
    for index, row in df.iterrows():
        url = row['Model URL']
        if pd.isna(url) or not url:
            print(f"Row {index + 1}: No URL found")
            continue
        
        print(f"\nProcessing row {index + 1}/{total_urls}: {url}")
        
        # Process the URL
        result = await process_url(url, config)
        
        # Update DataFrame with results
        if result.get("processing_status") == "success":
            successful += 1
            
            # Extract key information
            key_info = result.get("key_info", {})
            
            df.at[index, 'Title'] = result.get('title', '')
            df.at[index, 'Author'] = key_info.get('author', '')
            df.at[index, 'Publish_Date'] = key_info.get('publish_date', '')
            df.at[index, 'New_Price_Min'] = key_info.get('new_price_min', '')
            df.at[index, 'New_Price_Max'] = key_info.get('new_price_max', '')
            df.at[index, 'Used_Price_Min'] = key_info.get('used_price_min', '')
            df.at[index, 'Used_Price_Max'] = key_info.get('used_price_max', '')
            df.at[index, 'Road_Tax_Min'] = key_info.get('road_tax_min', '')
            df.at[index, 'Road_Tax_Max'] = key_info.get('road_tax_max', '')
            df.at[index, 'Insurance_Group'] = key_info.get('insurance_group', '')
            df.at[index, 'Fuel_Economy'] = key_info.get('fuel_economy', '')
            df.at[index, 'Range_Min'] = key_info.get('range_min', '')
            df.at[index, 'Range_Max'] = key_info.get('range_max', '')
            df.at[index, 'Doors'] = key_info.get('doors', '')
            df.at[index, 'Fuel_Types'] = key_info.get('fuel_types', '')
            df.at[index, 'Pros'] = key_info.get('pros', '')
            df.at[index, 'Cons'] = key_info.get('cons', '')
            df.at[index, 'Rivals'] = key_info.get('rivals', '')
            df.at[index, 'Article_Text'] = result.get('article_text', '')
            df.at[index, 'Full_JSON'] = json.dumps(result, indent=2)
            df.at[index, 'Processing_Status'] = 'success'
            df.at[index, 'Resolved_URL'] = result.get('url', '')
            
            print(f"✓ Successfully scraped: {result.get('title', 'N/A')}")
            
        else:
            failed += 1
            df.at[index, 'Processing_Status'] = 'failed'
            df.at[index, 'Full_JSON'] = json.dumps(result, indent=2)
            print(f"✗ Failed to scrape: {result.get('error', 'Unknown error')}")
    
    # Save updated Excel file
    output_path = excel_path.replace('.xlsx', '_enhanced.xlsx')
    df.to_excel(output_path, index=False)
    
    print(f"\n=== SCRAPING COMPLETE ===")
    print(f"Total URLs processed: {total_urls}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Parkers scraper for Series + URL.xlsx")
    parser.add_argument("--excel", default="Series + URL.xlsx", help="Input Excel file path")
    parser.add_argument("--output", help="Output Excel file path (default: input_with_data.xlsx)")
    parser.add_argument("--limit", type=int, help="Limit number of rows to process")
    parser.add_argument("--offset", type=int, default=0, help="Start from this row")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent tasks")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--debug-html", action="store_true", help="Save debug HTML files")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per page in seconds")
    
    args = parser.parse_args()
    
    config = ScrapeConfig(
        headless=args.headless,
        timeout=args.timeout * 1000, # Playwright timeout is in ms
        save_debug_html=args.debug_html
    )
    
    asyncio.run(process_excel_file(
        excel_path=args.excel,
        config=config,
        max_rows=args.limit
    ))

if __name__ == "__main__":
    main()
