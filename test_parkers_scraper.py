# Create a robust Playwright-based scraper that scrolls to load lazy content
# and extracts all text under the title, plus sectionized content.
script = r'''
import asyncio
import json
import re
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from playwright.async_api import async_playwright
from selectolax.parser import HTMLParser


COOKIES_BUTTON_SELECTORS = [
    "button#onetrust-accept-btn-handler",
    "button[aria-label='Accept all']",
    "button:has-text('Accept all cookies')",
    "button:has-text('Accept All Cookies')",
    "button:has-text('Accept')",
]

def normalize_ws(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

async def scroll_to_bottom(page, step: int = 700, delay_ms: int = 350, max_loops: int = 200):
    """
    Scrolls to the bottom of the page to trigger lazy loading.
    Stops when the document height stops increasing or after max_loops.
    """
    prev_height = -1
    loops = 0
    while loops < max_loops:
        curr_height = await page.evaluate("document.body.scrollHeight")
        if curr_height == prev_height:
            # Give a little extra nudge in case there are sentinels near bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(600)
            curr2 = await page.evaluate("document.body.scrollHeight")
            if curr2 == curr_height:
                break
        prev_height = curr_height
        await page.evaluate(f"window.scrollBy(0, {step});")
        await page.wait_for_timeout(delay_ms)
        loops += 1

def article_below_title(article_html: str) -> Dict[str, str]:
    """
    Return everything in <article> *after* the first <h1>.
    Also build a minimal markdown with h2->##, h3->###, list items, and paragraphs.
    """
    html = HTMLParser(article_html)
    # If no article content, return whole text
    if not html.body:
        t = html.text(separator="\n", strip=True)
        return {"article_text": t, "article_markdown": t}

    h1 = html.css_first("h1")
    if not h1:
        t = html.text(separator="\n", strip=True)
        return {"article_text": t, "article_markdown": t}

    texts: List[str] = []
    md_lines: List[str] = []

    node = h1.next
    while node:
        # Headings
        if node.tag == "h2":
            title = node.text(strip=True)
            if title:
                md_lines.append(f"## {title}")
        elif node.tag == "h3":
            title = node.text(strip=True)
            if title:
                md_lines.append(f"### {title}")
        # Lists
        elif node.tag in {"ul", "ol"}:
            for li in node.css("li"):
                li_txt = normalize_ws(li.text())
                if li_txt:
                    md_lines.append(f"- {li_txt}")
        # Paragraph-ish
        elif node.tag in {"p", "div"}:
            ptxt = normalize_ws(node.text())
            if ptxt:
                md_lines.append(ptxt)
        node = node.next

    md = "\n\n".join(md_lines).strip()
    # Plain text by stripping markdown into text
    plain = HTMLParser(f"<div>{md}</div>").text(separator="\n", strip=True)
    return {"article_text": plain, "article_markdown": md}

def extract_sections(article_html: str) -> Dict[str, str]:
    """
    Build a {section_title: text} map by walking h2 nodes and collecting following
    siblings until the next h2. Also returns a 'full_text' fallback.
    """
    html = HTMLParser(article_html)
    out: Dict[str, str] = {}
    out["full_text"] = html.text(separator="\n", strip=True)

    for h in html.css("h2"):
        title = h.text(strip=True)
        if not title:
            continue
        parts: List[str] = []
        node = h.next
        while node and node.tag != "h2":
            if node.tag in {"p", "div"}:
                t = normalize_ws(node.text())
                if t:
                    parts.append(t)
            elif node.tag in {"ul", "ol"}:
                for li in node.css("li"):
                    li_txt = normalize_ws(li.text())
                    if li_txt:
                        parts.append(f"- {li_txt}")
            node = node.next
        if parts:
            out[title] = "\n".join(parts)
    return out

async def scrape_parkers(url: str, save_html: Optional[Path] = None) -> Dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/114.0.0.0 Safari/537.36")
        )
        page = await ctx.new_page()
        await page.goto(url, timeout=60_000, wait_until="domcontentloaded")

        # Accept cookies if present
        for sel in COOKIES_BUTTON_SELECTORS:
            try:
                btn = await page.wait_for_selector(sel, timeout=2500)
                if btn:
                    await btn.click()
                    break
            except Exception:
                pass

        # Ensure article root exists
        await page.wait_for_selector("article", timeout=20_000)

        # Scroll to bottom to trigger lazy loading
        await scroll_to_bottom(page, step=800, delay_ms=350, max_loops=300)

        # Small buffer to let final blocks render
        await page.wait_for_timeout(800)

        rendered = await page.content()

        if save_html:
            save_html.write_text(rendered, encoding="utf-8")

        doc = HTMLParser(rendered)
        article = doc.css_first("article")

        if not article:
            title = doc.css_first("h1").text(strip=True) if doc.css_first("h1") else None
            return {
                "title": title,
                "article_text": None,
                "article_markdown": None,
                "sections": {},
                "url": url,
                "note": "No <article> found after scrolling"
            }

        title = article.css_first("h1").text(strip=True) if article.css_first("h1") else (
            doc.css_first("h1").text(strip=True) if doc.css_first("h1") else None
        )
        article_html = article.html

        below = article_below_title(article_html)
        sections = extract_sections(article_html)

        return {
            "title": title,
            "article_text": below["article_text"],
            "article_markdown": below["article_markdown"],
            "sections": sections,
            "url": url,
        }

def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape a Parkers review page (all text below the title).")
    parser.add_argument("url", nargs="?", default="https://www.parkers.co.uk/bmw/4-series/gran-coupe-2014/used-review/", help="Parkers review URL")
    parser.add_argument("--out-json", type=str, default="parkers_article.json", help="Save JSON to this path")
    parser.add_argument("--out-md", type=str, default="parkers_article.md", help="Save Markdown to this path")
    parser.add_argument("--save-html", type=str, default="parkers_debug.html", help="Save rendered HTML to this path")
    args = parser.parse_args()

    print(f"Scraping {args.url}...\n")
    data = asyncio.run(scrape_parkers(args.url, save_html=Path(args.save_html) if args.save_html else None))

    # Print brief preview
    print("=== TITLE ===\n", data.get("title") or "N/A")
    atxt = data.get("article_text") or ""
    print("\n=== ARTICLE TEXT (first 1200 chars) ===\n", atxt[:1200] or "N/A")
    print("\n=== H2 SECTIONS FOUND ===\n", [k for k in data.get("sections", {}).keys() if k != "full_text"])

    # Save outputs
    Path(args.out_json).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.out_md).write_text(f"# {data.get('title') or ''}\n\n{data.get('article_markdown') or ''}", encoding="utf-8")
    print(f"\nSaved rendered HTML -> {args.save_html}")
    print(f"Saved JSON -> {args.out_json}")
    print(f"Saved article markdown -> {args.out_md}")

if __name__ == "__main__":
    cli()
'''
from pathlib import Path

out_path = Path("/mnt/data/parkers_scrape_full.py")
out_path.write_text(script, encoding="utf-8")
out_path
