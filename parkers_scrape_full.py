import asyncio
import json
import re
import sys
from typing import Dict, List, Optional
from pathlib import Path

import httpx
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from selectolax.parser import HTMLParser

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/114.0.0.0 Safari/537.36")
}

COOKIES_BUTTON_SELECTORS = [
    "button#onetrust-accept-btn-handler",
    "button[aria-label='Accept all']",
    "button:has-text('Accept all cookies')",
    "button:has-text('Accept All Cookies')",
    "button:has-text('Accept')",
]

READ_MORE_SELECTORS = [
    "button:has-text('Read more')",
    "button:has-text('Show more')",
    "a:has-text('Read more')",
]

TAB_LIKE_SELECTORS = [
    "button:has-text('Overview')",
    "button:has-text('Practicality')",
    "button:has-text('Reliability')",
    "button:has-text('Running costs')",
    "button:has-text('Performance')",
    "button:has-text('Interior')",
    "button:has-text('Engines')",
    "a:has-text('Overview')",
    "a:has-text('Practicality')",
    "a:has-text('Reliability')",
    "a:has-text('Running costs')",
    "a:has-text('Performance')",
    "a:has-text('Interior')",
    "a:has-text('Engines')",
]

RE_YEAR_TAIL = re.compile(r"^(?P<body>[a-z0-9-]+)-(?:19|20)\d{2}$", re.I)

def normalize_ws(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_specs_url(url: str) -> bool:
    return url.rstrip("/").endswith("/specs")

async def http_exists(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.head(url, follow_redirects=True, timeout=10, headers=HEADERS)
        if r.status_code in (405, 403):
            r = await client.get(url, follow_redirects=True, timeout=10, headers=HEADERS)
        if 200 <= r.status_code < 300:
            return str(r.url)
        return None
    except Exception:
        return None

async def resolve_review_url(specs_url: str) -> Dict:
    u = specs_url.strip().split("?")[0].split("#")[0]
    parts = u.rstrip("/").split("/")
    if not parts or parts[-1] != "specs":
        return {"resolved_url": u, "status": "passthrough"}

    try:
        tail = parts[-2]
        model = parts[-3]
        brand = parts[-4]
    except Exception:
        return {"resolved_url": None, "status": "parse_error", "input": u}

    prefix = f"https://www.parkers.co.uk/{brand}/{model}/"

    candidates: List[str] = []
    def add(url: str):
        url = url.replace("//", "/").replace("https:/", "https://")
        candidates.append(url if url.endswith("/") else url + "/")
        candidates.append(url if not url.endswith("/") else url[:-1])

    add(prefix + tail + "/review")
    add(prefix + tail + "/used-review")

    m = RE_YEAR_TAIL.match(tail)
    if m:
        body = m.group("body")
        add(prefix + body + "/review")
        add(prefix + body + "/used-review")

    add(prefix + tail + "/owners-reviews")
    if m:
        add(prefix + body + "/owners-reviews")

    async with httpx.AsyncClient() as client:
        for i, cand in enumerate(candidates, start=1):
            final = await http_exists(client, cand)
            if final:
                return {"resolved_url": final, "status": "ok", "strategy_index": i, "candidate": cand}
    return {"resolved_url": None, "status": "not_found", "candidates_tried": candidates}

async def scroll_to_bottom(page, step: int = 900, delay_ms: int = 300, max_loops: int = 250):
    prev_height = -1
    loops = 0
    while loops < max_loops:
        curr_height = await page.evaluate("document.body.scrollHeight")
        if curr_height == prev_height:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(700)
            curr2 = await page.evaluate("document.body.scrollHeight")
            if curr2 == curr_height:
                break
        prev_height = curr_height
        await page.evaluate(f"window.scrollBy(0, {step});")
        await page.wait_for_timeout(delay_ms)
        loops += 1

async def safe_click_elements(page, selectors: List[str], max_clicks: int = 12, per_click_timeout_ms: int = 800):
    """
    Click only elements that won't navigate away; guard every click with a tiny timeout
    and use no_wait_after to avoid hanging on slow handlers.
    """
    clicks = 0
    for sel in selectors:
        try:
            loc = page.locator(sel)
            count = await loc.count()
            for i in range(count):
                if clicks >= max_clicks:
                    return clicks
                el = loc.nth(i)
                try:
                    href = await el.get_attribute("href")
                    # Skip links that navigate (href with http/https or non-fragment)
                    if href and href.startswith("http"):
                        continue
                    await asyncio.wait_for(
                        el.click(timeout=per_click_timeout_ms, no_wait_after=True),
                        timeout=per_click_timeout_ms / 1000 + 0.3
                    )
                    clicks += 1
                    await page.wait_for_timeout(150)
                except (PWTimeout, asyncio.TimeoutError):
                    continue
                except Exception:
                    continue
        except Exception:
            continue
    return clicks

def article_below_title(article_html: str) -> Dict[str, str]:
    html = HTMLParser(article_html)
    h1 = html.css_first("h1")
    if not h1:
        t = html.text(separator="\n", strip=True)
        return {"article_text": t, "article_markdown": t}

    md_lines: List[str] = []
    node = h1.next
    while node:
        if node.tag == "h2":
            title = node.text(strip=True)
            if title:
                md_lines.append(f"## {title}")
        elif node.tag == "h3":
            title = node.text(strip=True)
            if title:
                md_lines.append(f"### {title}")
        elif node.tag in {"ul", "ol"}:
            for li in node.css("li"):
                li_txt = normalize_ws(li.text())
                if li_txt:
                    md_lines.append(f"- {li_txt}")
        elif node.tag in {"p", "div"}:
            ptxt = normalize_ws(node.text())
            if ptxt:
                md_lines.append(ptxt)
        node = node.next

    md = "\n\n".join(md_lines).strip()
    plain = HTMLParser(f"<div>{md}</div>").text(separator="\n", strip=True)
    return {"article_text": plain, "article_markdown": md}

def extract_sections(article_html: str) -> Dict[str, str]:
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

async def scrape_parkers_review(url: str, save_html: Optional[Path] = None, expand: bool = False) -> Dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=HEADERS["User-Agent"])
        # Tighten default timeouts so nothing hangs forever
        ctx.set_default_timeout(8000)
        ctx.set_default_navigation_timeout(15000)

        page = await ctx.new_page()
        print("[goto]", url)
        await page.goto(url, timeout=60_000, wait_until="domcontentloaded")

        # Accept cookies if present
        for sel in COOKIES_BUTTON_SELECTORS:
            try:
                btn = await page.wait_for_selector(sel, timeout=2000)
                if btn:
                    await btn.click()
                    print("[cookies] accepted")
                    break
            except PWTimeout:
                pass
            except Exception:
                pass

        # Ensure article
        await page.wait_for_selector("article", timeout=15000)

        # Optional: safe tab/expander clicks (guarded)
        if expand:
            clicked_tabs = await safe_click_elements(page, TAB_LIKE_SELECTORS, max_clicks=10)
            if clicked_tabs:
                print(f"[expand] clicked {clicked_tabs} tab-like controls")
            clicked_more = await safe_click_elements(page, READ_MORE_SELECTORS, max_clicks=10)
            if clicked_more:
                print(f"[expand] clicked {clicked_more} read-more controls")

        # Scroll to the bottom to trigger lazy loading
        print("[scroll] start")
        await scroll_to_bottom(page, step=900, delay_ms=250, max_loops=240)
        print("[scroll] done")

        # One more pass for expanders that appeared
        if expand:
            clicked_more_2 = await safe_click_elements(page, READ_MORE_SELECTORS, max_clicks=8)
            if clicked_more_2:
                print(f"[expand] post-scroll clicked {clicked_more_2} more")

        await page.wait_for_timeout(700)
        rendered = await page.content()
        if save_html:
            save_html.write_text(rendered, encoding="utf-8")

        doc = HTMLParser(rendered)
        article = doc.css_first("article")
        if not article:
            title = (doc.css_first("h1").text(strip=True) if doc.css_first("h1") else None)
            return {"title": title, "article_text": None, "article_markdown": None,
                    "sections": {}, "url": url, "note": "No <article> found after scrolling"}

        title = article.css_first("h1").text(strip=True) if article.css_first("h1") else (
            doc.css_first("h1").text(strip=True) if doc.css_first("h1") else None
        )
        article_html = article.html

        below = article_below_title(article_html)
        sections = extract_sections(article_html)

        return {"title": title, "article_text": below["article_text"],
                "article_markdown": below["article_markdown"],
                "sections": sections, "url": url}

def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape Parkers review page (all text below the title).")
    parser.add_argument("url", help="Parkers /review(/used-review)/ or /specs/ URL")
    parser.add_argument("--out-json", default="parkers_article.json")
    parser.add_argument("--out-md", default="parkers_article.md")
    parser.add_argument("--save-html", default="parkers_debug.html")
    parser.add_argument("--expand", action="store_true",
                        help="Try to click tab/read-more elements (guarded). Off by default for stability.")
    args = parser.parse_args()

    url_in = args.url.strip()
    if is_specs_url(url_in):
        resolved = asyncio.run(resolve_review_url(url_in))
        if not resolved.get("resolved_url"):
            print("Could not resolve review URL from specs:", json.dumps(resolved, indent=2))
            sys.exit(2)
        review_url = resolved["resolved_url"]
        print(f"[resolved] {review_url}")
    else:
        review_url = url_in

    # Overall watchdog: if something blocks, you regain control
    async def run_with_watchdog():
        return await scrape_parkers_review(
            review_url, save_html=Path(args.save_html) if args.save_html else None, expand=args.expand
        )

    try:
        data = asyncio.run(asyncio.wait_for(run_with_watchdog(), timeout=75))
    except asyncio.TimeoutError:
        print("[timeout] overall scrape timed out at 75s")
        sys.exit(3)

    print("\n=== TITLE ===\n", data.get("title") or "N/A")
    atxt = data.get("article_text") or ""
    print("\n=== ARTICLE TEXT (first 1200 chars) ===\n", atxt[:1200] or "N/A")
    print("\n=== H2 SECTIONS FOUND ===\n", [k for k in data.get("sections", {}).keys() if k != "full_text"])

    Path(args.out_json).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.out_md).write_text(f"# {data.get('title') or ''}\n\n{data.get('article_markdown') or ''}", encoding="utf-8")
    print(f"\nSaved rendered HTML -> {args.save_html}")
    print(f"Saved JSON -> {args.out_json}")
    print(f"Saved article markdown -> {args.out_md}")

if __name__ == "__main__":
    cli()
