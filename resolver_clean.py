# resolver_clean.py
"""
Parkers URL resolver:
- Input: any Parkers URL (often /specs/ or /review/).
- Output: a working /review or /used-review URL (2xx), or a structured miss.

Public API:
    async def resolve_review_url(url_in: str) -> dict

Returns dict like:
{
  "ok": True|False,
  "review_url": "https://www.parkers.co.uk/.../used-review/",
  "status": "ok" | "fallback" | "overview_click" | "not_found_after_fallbacks" | ...,
  "tried": ["candidate1", "candidate2", ...]  # URLs that were probed
}

Dependencies:
    pip install httpx playwright
    playwright install
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, urljoin

import httpx
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# -----------------------------------------------------------------------------
# Config / Constants
# -----------------------------------------------------------------------------

PARKERS = "https://www.parkers.co.uk"
RE_YEAR_TAIL = re.compile(r"^(?P<body>[a-z0-9-]+)-(?:19|20)\d{2}$", re.I)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/114.0.0.0 Safari/537.36")
}

# Playwright selectors that commonly map to "Overview" on /specs/ pages
OVERVIEW_SELECTORS = [
    "a[role='tab']:has-text('Overview')",
    "button[role='tab']:has-text('Overview')",
    "a:has-text('Overview')",
    "button:has-text('Overview')",
]

COOKIE_BUTTON_SELECTORS = [
    "button#onetrust-accept-btn-handler",
    "button[aria-label='Accept all']",
    "button:has-text('Accept all cookies')",
    "button:has-text('Accept All Cookies')",
    "button:has-text('Accept')",
]

# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------

def _normalize(u: str) -> str:
    """
    Lowercase path, strip query/fragment, collapse //, ensure https host.
    Keep no trailing slash to simplify equals-checks; we append / when probing.
    """
    s = urlsplit(u.strip())
    scheme = s.scheme or "https"
    netloc = s.netloc or "www.parkers.co.uk"
    path = re.sub(r"//+", "/", s.path.lower()).rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))

def _is_specs(u: str) -> bool:
    return _normalize(u).endswith("/specs")

def _is_reviewish(u: str) -> bool:
    uu = _normalize(u)
    return uu.endswith("/review") or uu.endswith("/used-review") or uu.endswith("/owners-reviews")

# -----------------------------------------------------------------------------
# HTTP probing
# -----------------------------------------------------------------------------

async def _exists(client: httpx.AsyncClient, url: str) -> Optional[str]:
    """
    Return final URL (string) if URL responds 2xx (after redirects), else None.
    Uses HEAD first; falls back to GET on 405/403/400.
    """
    url = _normalize(url)
    try:
        r = await client.head(url + "/", headers=HEADERS, follow_redirects=True, timeout=10)
        if r.status_code in (405, 403, 400):
            r = await client.get(url + "/", headers=HEADERS, follow_redirects=True, timeout=10)
        if 200 <= r.status_code < 300:
            return str(r.url)
        return None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Playwright-based resolution
# -----------------------------------------------------------------------------

async def resolve_via_specs_tabs_playwright(specs_url: str, timeout_s: int = 20) -> Optional[str]:
    """
    Open the /specs page, click the 'Review' tab if necessary, and resolve the canonical /review/ URL.
    Logs detailed progress to help debug why review resolution fails.
    """
    print(f"\n[Resolver] Starting resolve_via_specs_tabs_playwright for: {specs_url}")
    print(f"[Resolver] Timeout set to: {timeout_s} seconds")

    try:
        print("[Resolver] Launching Playwright...")
        async with async_playwright() as p:
            print("[Resolver] Launching Chromium browser...")
            browser = await p.chromium.launch(headless=False)  # Set to False so you can see actions
            print("[Resolver] Creating browser context...")
            ctx = await browser.new_context(user_agent=HEADERS["User-Agent"])
            print("[Resolver] Creating new page...")
            page = await ctx.new_page()
            print("[Resolver] Navigating to specs page...")
            await page.goto(specs_url, timeout=60_000, wait_until="domcontentloaded")
            print("[Resolver] Page loaded successfully!")

            # Accept cookies if present
            print("[Resolver] Checking for cookie buttons...")
            for sel in COOKIE_BUTTON_SELECTORS:
                try:
                    print(f"[Resolver] Trying cookie selector: {sel}")
                    btn = await page.wait_for_selector(sel, timeout=1500)
                    if btn:
                        await btn.click()
                        print("[Resolver] Accepted cookies")
                        break
                except Exception as e:
                    print(f"[Resolver] Cookie selector {sel} failed: {e}")
                    pass

            # Locate Review tab
            print("[Resolver] Searching for Review tab...")
            # Look for Review tabs within the specs page content, not global navigation
            review_tab = page.locator("main a:has-text('Review'), main button:has-text('Review'), .specs-content a:has-text('Review'), .specs-content button:has-text('Review'), [role='tab']:has-text('Review')")
            count = await review_tab.count()
            print(f"[Resolver] Review tab count = {count}")

            if count == 0:
                print("[Resolver] No Review tab found → fallback to specs.")
                await browser.close()
                return _normalize(specs_url)

            # Check if the tab has a valid href
            href = await review_tab.first.get_attribute("href")
            print(f"[Resolver] Review tab href = {href}")

            if href and href != "#" and not href.lower().startswith("javascript"):
                resolved = urljoin(specs_url, href)
                print(f"[Resolver] Found direct review href: {resolved}")
                # Only accept if it's a specific car review, not a generic reviews page
                if _is_reviewish(resolved) and not resolved.endswith("/car-reviews/"):
                    print(f"[Resolver] Found specific car review href: {resolved}")
                    await browser.close()
                    return _normalize(resolved)
                else:
                    print(f"[Resolver] Review href is generic, will click tab instead: {resolved}")

            # Otherwise, click the tab
            print("[Resolver] Clicking Review tab...")
            await review_tab.first.scroll_into_view_if_needed()
            await review_tab.first.click()

            # Wait for review-specific content to load
            print("[Resolver] Waiting for review content...")
            try:
                await page.wait_for_selector(
                    "h2:has-text('Pros & cons'), h2:has-text('Verdict'), h2:has-text('Overview')",
                    timeout=timeout_s * 1000
                )
                print("[Resolver] Review content loaded!")
            except Exception as e:
                print(f"[Resolver] Timed out waiting for review content: {e} → fallback to specs.")
                await browser.close()
                return _normalize(specs_url)

            # Try canonical URL first
            print("[Resolver] Looking for canonical link...")
            try:
                canonical = await page.locator("head link[rel='canonical']").get_attribute("href")
                print(f"[Resolver] Canonical href = {canonical}")
                if canonical and _is_reviewish(canonical):
                    print(f"[Resolver] Resolved review via canonical: {canonical}")
                    await browser.close()
                    return _normalize(canonical)
            except Exception as e:
                print(f"[Resolver] No canonical link found: {e}")

            # Fallback: scan for any review links on the page
            print("[Resolver] Scanning page for review links...")
            anchors = page.locator("a[href*='/review'], a[href*='/used-review']")
            if await anchors.count() > 0:
                href = await anchors.first.get_attribute("href")
                if href and _is_reviewish(href):
                    resolved = urljoin(specs_url, href)
                    print(f"[Resolver] Found review link in page content: {resolved}")
                    await browser.close()
                    return _normalize(resolved)

            # If nothing found, return specs as a safe fallback
            print("[Resolver] Could not resolve review → returning specs URL.")
            await browser.close()
            return _normalize(specs_url)

    except Exception as e:
        print(f"[Resolver] Unexpected error: {e}")
        import traceback
        print(f"[Resolver] Full traceback: {traceback.format_exc()}")
        return _normalize(specs_url)

# -----------------------------------------------------------------------------
# Main resolver function
# -----------------------------------------------------------------------------

async def resolve_review_url(url_in: str) -> Dict:
    """
    Main resolver function that tries multiple strategies to find a working review URL.
    """
    print(f"[Resolver] Starting resolution for: {url_in}")
    
    # For now, just try the Playwright approach directly
    result = await resolve_via_specs_tabs_playwright(url_in)
    
    if result and _is_reviewish(result):
        return {
            "ok": True,
            "review_url": result,
            "status": "playwright_resolved",
            "tried": [url_in, result]
        }
    else:
        return {
            "ok": False,
            "review_url": result,
            "status": "fallback_to_specs",
            "tried": [url_in, result]
        }
