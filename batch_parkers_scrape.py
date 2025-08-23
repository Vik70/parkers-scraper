import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from resolver import resolve_review_url


import httpx
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from selectolax.parser import HTMLParser

# ==============================
# Config & Helpers
# ==============================

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

# Optional “tab-like” buttons. We keep this OFF by default for stability.
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

def slugify(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip())
    return s.lower()[:max_len] or "row"

def is_specs_url(url: str) -> bool:
    return url.rstrip("/").endswith("/specs")

async def http_exists(client: httpx.AsyncClient, url: str) -> Optional[str]:
    """Return final URL if 2xx after redirects, else None."""
    try:
        r = await client.head(url, follow_redirects=True, timeout=10, headers=HEADERS)
        if r.status_code in (405, 403, 400):  # some servers dislike HEAD
            r = await client.get(url, follow_redirects=True, timeout=10, headers=HEADERS)
        if 200 <= r.status_code < 300:
            return str(r.url)
        return None
    except Exception:
        return None

async def resolve_review_url(specs_url: str) -> Dict:
    """
    Resolve /specs/ → review or used-review based on common Parkers patterns.
    Tries a handful of candidate URLs; returns the first that exists (2xx).
    """
    u = specs_url.strip().split("?")[0].split("#")[0]
    parts = u.rstrip("/").split("/")
    if not parts or parts[-1] != "specs":
        return {"resolved_url": u, "status": "passthrough"}

    try:
        tail = parts[-2]       # body[-year]
        model = parts[-3]      # model
        brand = parts[-4]      # brand
    except Exception:
        return {"resolved_url": None, "status": "parse_error", "input": u}

    prefix = f"https://www.parkers.co.uk/{brand}/{model}/"

    candidates: List[str] = []
    def add(url: str):
        url = url.replace("//", "/").replace("https:/", "https://")
        candidates.append(url if url.endswith("/") else url + "/")
        candidates.append(url if not url.endswith("/") else url[:-1])

    # with tail as-is
    add(prefix + tail + "/review")
    add(prefix + tail + "/used-review")

    # drop trailing year if present
    m = RE_YEAR_TAIL.match(tail)
    if m:
        body = m.group("body")
        add(prefix + body + "/review")
        add(prefix + body + "/used-review")

    # owner reviews fallback
    add(prefix + tail + "/owners-reviews")
    if m:
        add(prefix + body + "/owners-reviews")

    async with httpx.AsyncClient() as client:
        for i, cand in enumerate(candidates, start=1):
            final = await http_exists(client, cand)
            if final:
                return {"resolved_url": final, "status": "ok", "strategy_index": i, "candidate": cand}

    return {"resolved_url": None, "status": "not_found", "candidates_tried": candidates}

async def scroll_to_bottom(page, step: int = 900, delay_ms: int = 250, max_loops: int = 240):
    """Scroll to bottom to trigger lazy-loading; stops when height stops changing."""
    prev_height = -1
    loops = 0
    while loops < max_loops:
        curr_height = await page.evaluate("document.body.scrollHeight")
        if curr_height == prev_height:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(600)
            curr2 = await page.evaluate("document.body.scrollHeight")
            if curr2 == curr_height:
                break
        prev_height = curr_height
        await page.evaluate(f"window.scrollBy(0, {step});")
        await page.wait_for_timeout(delay_ms)
        loops += 1

async def safe_click_elements(page, selectors: List[str], max_clicks: int = 10, per_click_timeout_ms: int = 700):
    """Safely click expandable controls (no navigation), guarded by small timeouts."""
    from playwright.async_api import TimeoutError as PWTimeout
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
                    if href and href.startswith("http"):
                        continue  # avoid navigation
                    await asyncio.wait_for(
                        el.click(timeout=per_click_timeout_ms, no_wait_after=True),
                        timeout=per_click_timeout_ms/1000 + 0.3
                    )
                    clicks += 1
                    await page.wait_for_timeout(120)
                except (PWTimeout, asyncio.TimeoutError):
                    continue
                except Exception:
                    continue
        except Exception:
            continue
    return clicks

def article_below_title(article_html: str) -> Dict[str, str]:
    """Everything in <article> after the first <h1> (markdown + plain text)."""
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
    """Map {h2: text} plus a 'full_text' fallback covering the whole article."""
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

@dataclass
class ScrapeConfig:
    expand: bool = False
    save_html: Optional[Path] = None
    overall_timeout_s: int = 75

async def scrape_review_page(url: str, cfg: ScrapeConfig) -> Dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=HEADERS["User-Agent"])
        ctx.set_default_timeout(8000)
        ctx.set_default_navigation_timeout(15000)
        page = await ctx.new_page()

        await page.goto(url, timeout=60_000, wait_until="domcontentloaded")

        # accept cookies
        for sel in COOKIES_BUTTON_SELECTORS:
            try:
                btn = await page.wait_for_selector(sel, timeout=2000)
                if btn:
                    await btn.click()
                    break
            except PWTimeout:
                pass
            except Exception:
                pass

        await page.wait_for_selector("article", timeout=15000)

        if cfg.expand:
            await safe_click_elements(page, TAB_LIKE_SELECTORS, max_clicks=10)
            await safe_click_elements(page, READ_MORE_SELECTORS, max_clicks=10)

        await scroll_to_bottom(page, step=900, delay_ms=250, max_loops=240)
        if cfg.expand:
            await safe_click_elements(page, READ_MORE_SELECTORS, max_clicks=6)

        await page.wait_for_timeout(600)
        rendered = await page.content()
        if cfg.save_html:
            cfg.save_html.write_text(rendered, encoding="utf-8")

        doc = HTMLParser(rendered)
        article = doc.css_first("article")
        if not article:
            title = (doc.css_first("h1").text(strip=True) if doc.css_first("h1") else None)
            return {"title": title, "article_text": None, "article_markdown": None,
                    "sections": {}, "url": url, "note": "No <article> found"}

        title = article.css_first("h1").text(strip=True) if article.css_first("h1") else (
            doc.css_first("h1").text(strip=True) if doc.css_first("h1") else None
        )
        article_html = article.html

        below = article_below_title(article_html)
        sections = extract_sections(article_html)

        return {"title": title, "article_text": below["article_text"],
                "article_markdown": below["article_markdown"],
                "sections": sections, "url": url}

async def scrape_one(
    url: str,
    out_dir: Path,
    meta: Dict,
    expand: bool,
    retries: int = 2
) -> Dict:
    """
    Resolve ANY Parkers URL to a valid review/used-review page first,
    scrape it, and write JSON. If nothing resolves, skip scraping and
    write a stub JSON instead.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build safe filename
    base = meta.get("id") or meta.get("slug") or slugify(meta.get("name") or meta.get("model") or url)
    fname = slugify(base)
    h = abs(hash(url)) % (10**8)
    json_path = out_dir / f"{fname}-{h}.json"

    status = {"url_in": url, "json_path": str(json_path), **meta}

    # ALWAYS use resolver first
    try:
        pre = await resolve_review_url(url)
        status["resolved_status"] = pre.get("status")
        status["review_url"] = pre.get("review_url")
        status["candidates_tried"] = "|".join(pre.get("tried", []))
    except Exception as e:
        status.update({"ok": False, "error": f"resolver_failed: {e}"})
        return status

    # If no valid URL found, save stub JSON + stop
    if not pre.get("ok"):
        json_path.write_text(json.dumps({
            "title": "Preflight failed",
            "article_text": "",
            "article_markdown": "",
            "sections": {"full_text": ""},
            "url": url,
            "review_url": pre.get("review_url"),
            "preflight_status": pre.get("status"),
            "candidates_tried": pre.get("tried", []),
            "meta": meta
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        status["ok"] = False
        status["error"] = pre.get("status") or "could_not_resolve_review_url"
        return status

    review_url = pre["review_url"]

    # Attempt scraping resolved URL
    attempt = 0
    last_err = None
    while attempt <= retries:
        try:
            cfg = ScrapeConfig(expand=expand, save_html=None, overall_timeout_s=75)
            data = await asyncio.wait_for(scrape_review_page(review_url, cfg), timeout=cfg.overall_timeout_s)
            data["meta"] = meta
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            status["ok"] = True
            status["error"] = ""
            return status
        except asyncio.TimeoutError as e:
            last_err = f"timeout: {e}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        attempt += 1
        await asyncio.sleep(1.0 * attempt)

    # Scrape failed after retries
    status["ok"] = False
    status["error"] = last_err or "unknown_error"
    return status


# ==============================
# Batch runner
# ==============================

def autodetect_url_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "url" in str(c).lower():
            return c
    return None

async def run_batch(
    excel_path: Path,
    out_dir: Path,
    sheet: Optional[str],
    url_col: Optional[str],
    id_cols: List[str],
    limit: Optional[int],
    offset: int,
    concurrency: int,
    expand: bool,
    index_csv: Path,
):
    df = pd.read_excel(excel_path, sheet_name=sheet or 0)
    if url_col is None:
        url_col = autodetect_url_column(df)
    if url_col is None:
        raise ValueError("Could not auto-detect URL column. Use --url-col to specify.")

    if limit is not None:
        df = df.iloc[offset: offset + limit]
    elif offset:
        df = df.iloc[offset:]

    # Build metadata per row
    tasks = []
    sem = asyncio.Semaphore(concurrency)

    rows_meta = []
    for i, row in df.iterrows():
        url = str(row[url_col]).strip()
        if not url or url == "nan":
            continue
        meta = {}
        for c in id_cols:
            if c in df.columns:
                meta[c] = row[c]
        # nice defaults
        meta["id"] = meta.get("id") or row.get("ID") or row.get("Id") or f"row{i}"
        meta["name"] = meta.get("name") or row.get("Name") or row.get("Series") or row.get("Model")
        meta["model"] = meta.get("model") or row.get("Model")
        rows_meta.append((url, meta))

    results: List[Dict] = []

    async def worker(url: str, meta: Dict):
        async with sem:
            status = await scrape_one(url, out_dir, meta, expand=expand, retries=2)
            results.append(status)

    await asyncio.gather(*(worker(url, meta) for url, meta in rows_meta))

    # write index CSV
    pd.DataFrame(results).to_csv(index_csv, index=False)
    print(f"\nWrote index CSV -> {index_csv}")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Batch scrape Parkers URLs from an Excel sheet and output JSON files.")
    p.add_argument("--excel", required=True, help="Path to Excel file (e.g., 'Series + URL.xlsx')")
    p.add_argument("--sheet", default=None, help="Sheet name or index; defaults to first sheet")
    p.add_argument("--url-col", default=None, help="Name of the URL column (auto-detects if omitted)")
    p.add_argument("--id-cols", nargs="*", default=["ID", "Series", "Make", "Model"], help="Columns to include as metadata")
    p.add_argument("--out-dir", default="out_json", help="Output directory for JSON files")
    p.add_argument("--index-csv", default="scrape_index.csv", help="Where to write a run index CSV")
    p.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    p.add_argument("--offset", type=int, default=0, help="Start from this row offset")
    p.add_argument("--concurrency", type=int, default=3, help="Concurrent browser tasks (3–5 recommended)")
    p.add_argument("--expand", action="store_true", help="Click tab/read-more controls (guarded). Off by default.")
    args = p.parse_args()

    excel_path = Path(args.excel).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    index_csv = Path(args.index_csv).expanduser()

    asyncio.run(run_batch(
        excel_path=excel_path,
        out_dir=out_dir,
        sheet=args.sheet,
        url_col=args.url_col,
        id_cols=args.id_cols,
        limit=args.limit,
        offset=args.offset,
        concurrency=args.concurrency,
        expand=args.expand,
        index_csv=index_csv,
    ))

if __name__ == "__main__":
    main()
