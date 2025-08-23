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
    timeout_s: int = 60
    save_debug_html: bool = False
    retry_count: int = 2

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

def extract_sections_from_html(html_content: str, url: str) -> Dict[str, Any]:
    """
    Extract structured content from Parkers review page HTML.
    Returns data in the same format as parkers_article.json.
    """
    tree = HTMLParser(html_content)
    
    # Extract title
    title_elem = tree.css_first("h1")
    title = normalize_ws(title_elem.text()) if title_elem else "Unknown Review"
    
    # Extract main content areas
    sections = {}
    full_text_parts = []
    
    # Get all text content
    body = tree.css_first("body")
    if body:
        full_text_parts.append(normalize_ws(body.text()))
    
    # Look for specific review sections using simple CSS selectors
    section_selectors = {
        "At a glance": [".at-a-glance", ".glance-section", "[class*='glance']"],
        "Pros & cons": [".pros-cons", "[class*='pros']", "[class*='cons']"],
        "Overview": [".overview", "[class*='overview']"],
        "Verdict": [".verdict", "[class*='verdict']"],
        "Practicality & safety": ["[class*='practicality']", "[class*='safety']"],
        "Interior, tech & comfort": ["[class*='interior']", "[class*='comfort']"],
        "Engines & handling": ["[class*='engines']", "[class*='handling']", "[class*='performance']"],
        "Ownership cost": ["[class*='ownership']", "[class*='cost']"],
        "About the author": [".author", ".by-author", "[class*='author']"],
    }
    
    # Also look for headings with relevant text
    headings = tree.css("h1, h2, h3, h4, h5, h6")
    for heading in headings:
        heading_text = normalize_ws(heading.text()).lower()
        # Try to match heading text to sections
        if "pros" in heading_text or "cons" in heading_text:
            # Get the content after this heading
            next_content = get_content_after_element(heading, tree)
            if next_content:
                sections["Pros & cons"] = next_content
        elif "overview" in heading_text:
            next_content = get_content_after_element(heading, tree)
            if next_content:
                sections["Overview"] = next_content
        elif "verdict" in heading_text:
            next_content = get_content_after_element(heading, tree)
            if next_content:
                sections["Verdict"] = next_content
        elif "practicality" in heading_text or "safety" in heading_text:
            next_content = get_content_after_element(heading, tree)
            if next_content:
                sections["Practicality & safety"] = next_content
        elif "interior" in heading_text or "comfort" in heading_text:
            next_content = get_content_after_element(heading, tree)
            if next_content:
                sections["Interior, tech & comfort"] = next_content
        elif "engine" in heading_text or "handling" in heading_text or "performance" in heading_text:
            next_content = get_content_after_element(heading, tree)
            if next_content:
                sections["Engines & handling"] = next_content
        elif "ownership" in heading_text or "cost" in heading_text:
            next_content = get_content_after_element(heading, tree)
            if next_content:
                sections["Ownership cost"] = next_content
        elif "author" in heading_text:
            next_content = get_content_after_element(heading, tree)
            if next_content:
                sections["About the author"] = next_content
    
    # Try to extract specific sections using class-based selectors
    for section_name, selectors in section_selectors.items():
        if section_name in sections:  # Skip if already found via headings
            continue
        for selector in selectors:
            elements = tree.css(selector)
            if elements:
                # Get text from the first matching element and its siblings
                element = elements[0]
                section_text = normalize_ws(element.text())
                if section_text and len(section_text) > 10:  # Only use substantial content
                    sections[section_name] = section_text
                    break
    
    # Extract rivals section
    rivals_elements = tree.css(".rivals, [class*='rivals'], [class*='competitor']")
    if not rivals_elements:
        # Look for rivals mentioned in headings
        for heading in tree.css("h1, h2, h3, h4, h5, h6"):
            heading_text = normalize_ws(heading.text()).lower()
            if "rivals" in heading_text or "competitor" in heading_text:
                rivals_content = get_content_after_element(heading, tree)
                if rivals_content:
                    brand_model = title.split("(")[0].strip()
                    sections[f"{brand_model} rivals"] = rivals_content
                break
    else:
        rivals_text = normalize_ws(rivals_elements[0].text())
        if rivals_text:
            # Clean up the title to match the pattern
            brand_model = title.split("(")[0].strip()
            sections[f"{brand_model} rivals"] = rivals_text
    
    # Extract all paragraph content as fallback
    paragraphs = tree.css("p")
    article_text_parts = []
    for p in paragraphs[:20]:  # Limit to first 20 paragraphs to avoid too much text
        text = normalize_ws(p.text())
        if text and len(text) > 20:  # Only substantial paragraphs
            article_text_parts.append(text)
    
    article_text = " ".join(article_text_parts)
    full_text = " ".join(full_text_parts)
    
    # Create the result structure matching parkers_article.json
    result = {
        "title": title,
        "article_text": article_text,
        "article_markdown": article_text,  # For now, same as article_text
        "sections": {
            "full_text": full_text,
            **sections
        },
        "url": url
    }
    
    return result

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
            await page.goto(url, timeout=config.timeout_s * 1000, wait_until="domcontentloaded")
            
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
            return extract_sections_from_html(html_content, url)
            
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

async def process_excel_file(
    excel_path: str,
    output_excel_path: str = None,
    limit: Optional[int] = None,
    offset: int = 0,
    concurrency: int = 3,
    config: ScrapeConfig = None
):
    """
    Process the Excel file:
    1. Read Series + URL.xlsx
    2. Process each URL and get JSON data  
    3. Populate column C with JSON data
    4. Save updated Excel file
    """
    if config is None:
        config = ScrapeConfig()
    
    if output_excel_path is None:
        output_excel_path = excel_path.replace('.xlsx', '_with_data.xlsx')
    
    print(f"[Excel] Reading file: {excel_path}")
    
    # Read the Excel file
    df = pd.read_excel(excel_path)
    print(f"[Excel] Found {len(df)} rows")
    
    # Apply limit and offset
    if offset > 0:
        df = df.iloc[offset:]
    if limit is not None:
        df = df.iloc[:limit]
    
    print(f"[Excel] Processing {len(df)} rows (offset={offset}, limit={limit})")
    
    # Add column C for JSON data if it doesn't exist
    if len(df.columns) < 3:
        df['JSON_Data'] = None
    else:
        # Use existing third column
        df.iloc[:, 2] = None
    
    # Semaphore for concurrency control
    sem = asyncio.Semaphore(concurrency)
    
    async def process_row(idx, row):
        async with sem:
            url = str(row.iloc[1]).strip()  # Column B (Model URL)
            if not url or url == 'nan':
                return idx, {"error": "No URL"}
            
            try:
                json_data = await process_single_url(url, config)
                return idx, json_data
            except Exception as e:
                print(f"[Excel] Error processing row {idx}: {e}")
                return idx, {"error": str(e)}
    
    # Process all rows concurrently
    tasks = [process_row(idx, row) for idx, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    
    # Update DataFrame with results
    for idx, json_data in results:
        df.iloc[idx, 2] = json.dumps(json_data, ensure_ascii=False)
    
    # Save updated Excel file
    print(f"[Excel] Saving updated file to: {output_excel_path}")
    df.to_excel(output_excel_path, index=False)
    
    print(f"[Excel] Complete! Processed {len(results)} rows")
    return output_excel_path

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
        timeout_s=args.timeout,
        save_debug_html=args.debug_html
    )
    
    asyncio.run(process_excel_file(
        excel_path=args.excel,
        output_excel_path=args.output,
        limit=args.limit,
        offset=args.offset,
        concurrency=args.concurrency,
        config=config
    ))

if __name__ == "__main__":
    main()
