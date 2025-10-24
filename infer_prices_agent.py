#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import math
import random
import urllib.parse as _url

import requests
from bs4 import BeautifulSoup

import pandas as pd


def load_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    env = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env):
        with open(env, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.split("=", 1)[0].strip() == "OPENAI_API_KEY":
                    os.environ["OPENAI_API_KEY"] = s.split("=", 1)[1].strip().strip('"').strip("'")
                    break


def normalize_year(val: Any) -> str:
    s = str(val or "").strip()
    if not s:
        return ""
    s_low = s.lower()
    if s_low in {"now", "present", "current"}:
        return str(pd.Timestamp.now().year)
    return s


def family_key(row: pd.Series) -> str:
    parts = [
        str(row.get("Make", "")).strip().lower(),
        str(row.get("Model", "")).strip().lower(),
        str(row.get("Series (production years start-end)", "")).strip().lower(),
        normalize_year(row.get("Year start", "")),
        normalize_year(row.get("Year end", "")),
        str(row.get("engine type", "")).strip().lower(),
    ]
    return "|".join(parts)


SYSTEM = (
    "You are an expert UK used-car pricing analyst. Given a car's make, model, series/years and basic specs,"
    " infer realistic UK used price range in GBP for typical examples on the market (not outliers). Return strict JSON."
)


def build_prompt(meta: Dict[str, str]) -> str:
    return (
        "Infer UK used price range in GBP for this car.\n"
        "Rules:\n"
        "- Use current UK market context (assume normal mileage/condition).\n"
        "- Return conservative min/max reflecting typical listings, not the absolute extremes.\n"
        "- If the series spans many years, give a single representative range for the family.\n"
        "- If data is sparse, use best judgement and similar models.\n"
        "- UK currency only.\n\n"
        f"Make: {meta.get('Make','').strip()}\n"
        f"Model: {meta.get('Model','').strip()}\n"
        f"Series/Years: {meta.get('Series','').strip()} | start={meta.get('YearStart','')} end={meta.get('YearEnd','')}\n"
        f"Body: {meta.get('Body','')} | Engine: {meta.get('Engine','')} | Power(bhp): {meta.get('Power','')} | Trans: {meta.get('Trans','')}\n"
        f"Doors: {meta.get('Doors','')} | Seats: {meta.get('Seats','')}\n\n"
        "Return JSON only with keys: min_used_price_gbp (number), max_used_price_gbp (number), confidence ('high'|'medium'|'low'), notes (short)."
    )


def call_llm(client, model: str, meta: Dict[str, str]) -> Optional[Dict[str, Any]]:
    user = build_prompt(meta)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None
    # extract JSON
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            data = json.loads(m.group(0))
            return data
        except Exception:
            return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_prices_from_html(html: str) -> List[float]:
    soup = BeautifulSoup(html, "html.parser")
    prices: List[float] = []
    # Common Autotrader selectors fallbacks
    selectors = [
        '[data-testid="search-listing-price"]',
        '[data-testid="advert-price"]',
        '.vehicle-price',
        '.product-card-pricing__price',
    ]
    for sel in selectors:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            m = re.search(r"£\s*([\d,]+)", txt)
            if m:
                try:
                    prices.append(float(m.group(1).replace(",", "")))
                except Exception:
                    pass
    # Fallback: scan all text chunks for £
    if not prices:
        for el in soup.find_all(text=re.compile("£")):
            txt = str(el)
            m = re.search(r"£\s*([\d,]+)", txt)
            if m:
                try:
                    prices.append(float(m.group(1).replace(",", "")))
                except Exception:
                    pass
    # Deduplicate
    prices = list(dict.fromkeys(prices))
    return prices


def fetch_autotrader_prices(meta: Dict[str, str], max_pages: int, timeout: int, postcode: str) -> Optional[Dict[str, Any]]:
    make = (meta.get("Make") or "").strip()
    model = (meta.get("Model") or "").strip()
    year_from = normalize_year(meta.get("YearStart"))
    year_to = normalize_year(meta.get("YearEnd"))
    if not make or not model:
        return None

    prices: List[float] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        )
    }
    for page in range(1, max_pages + 1):
        q = {
            "sort": "price-asc",
            "postcode": postcode or "SW1A 0AA",
            "radius": "1500",
            "include-delivery-option": "on",
            "make": make,
            "model": model,
            "page": str(page),
        }
        if year_from:
            q["year-from"] = year_from
        if year_to:
            q["year-to"] = year_to
        url = "https://www.autotrader.co.uk/car-search?" + _url.urlencode(q, doseq=True)
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                continue
            prices.extend(_parse_prices_from_html(r.text))
        except Exception:
            continue

    if not prices:
        return None

    prices = [p for p in prices if p >= 100]  # basic sanity
    prices.sort()
    if not prices:
        return None
    # Trim extremes (10% each side) if enough samples
    if len(prices) >= 20:
        lo_idx = int(math.floor(len(prices) * 0.1))
        hi_idx = int(math.ceil(len(prices) * 0.9)) - 1
        lo = prices[lo_idx]
        hi = prices[hi_idx]
    else:
        lo, hi = prices[0], prices[-1]
    # Build payload consistent with LLM path
    return {
        "min_used_price_gbp": lo,
        "max_used_price_gbp": hi,
        "confidence": "medium",
        "notes": f"autotrader sample n={len(prices)} pages={max_pages}",
    }


def sanitize_prices(data: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], str, str]:
    if not data or not isinstance(data, dict):
        return None, None, "low", "no-parse"
    def to_num(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None
    lo = to_num(data.get("min_used_price_gbp"))
    hi = to_num(data.get("max_used_price_gbp"))
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    conf = str(data.get("confidence", "")) or ""
    notes = str(data.get("notes", "")) or ""
    # sanity bounds to avoid absurdities
    if lo is not None and lo < 100:  # almost no UK used cars < £100
        lo = 100.0
    if hi is not None and hi < (lo or 100):
        hi = float(lo or 100)
    return lo, hi, conf or "medium", notes


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer used price min/max (GBP) for all cars using LLM")
    ap.add_argument("--input", default="database_2.0_enriched.xlsx")
    ap.add_argument("--sheet", default="Sheet2")
    ap.add_argument("--output", default="database_2.0_enriched_prices.xlsx")
    ap.add_argument("--cache", default="price_cache.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--source", choices=["autotrader", "llm", "hybrid"], default="autotrader")
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--postcode", default="SW1A 0AA")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recompute all families, ignore cache")
    ap.add_argument("--only-missing", action="store_true", help="Only compute families where price columns are blank")
    args = ap.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet, dtype=str)
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    # Build per-family metadata and cache keys
    df["__fid"] = df.apply(family_key, axis=1)

    # Load cache
    cache: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.cache):
        with open(args.cache, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and "fid" in rec and "data" in rec:
                        cache[rec["fid"]] = rec["data"]
                except Exception:
                    pass

    # families to process
    fams: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        fid = r["__fid"]
        if fid in fams:
            continue
        fams[fid] = {
            "Make": str(r.get("Make", "")),
            "Model": str(r.get("Model", "")),
            "Series": str(r.get("Series (production years start-end)", "")),
            "YearStart": normalize_year(r.get("Year start", "")),
            "YearEnd": normalize_year(r.get("Year end", "")),
            "Body": str(r.get("Real Body Type", "")),
            "Engine": str(r.get("engine type", "")),
            "Power": str(r.get("Power (bhp)", "")),
            "Trans": str(r.get("Transmission", "")),
            "Doors": str(r.get("Doors", "")),
            "Seats": str(r.get("Seats", "")),
        }

    # Prepare client (for LLM or hybrid fallback)
    client = None
    if args.source in {"llm", "hybrid"}:
        from openai import OpenAI
        load_api_key()
        client = OpenAI()

    # Worklist
    # Decide which families to (re)compute
    if args.force:
        to_run = list(fams.keys())
        cache.clear()
        if os.path.exists(args.cache):
            try:
                os.remove(args.cache)
            except Exception:
                pass
    elif args.only_missing:
        missing_fids = set()
        for i, r in df.iterrows():
            if (str(r.get("Min. of used price low helper", "")).strip() == "" or
                str(r.get("Max. of used price high helper", "")).strip() == ""):
                missing_fids.add(r["__fid"])
        to_run = [fid for fid in fams.keys() if (fid in missing_fids) and (fid not in cache)]
    else:
        to_run = [fid for fid in fams.keys() if fid not in cache]
    results: Dict[str, Dict[str, Any]] = {}

    if to_run:
        start = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex, open(args.cache, "a", encoding="utf-8") as cf:
            def runner(fid: str):
                meta = fams[fid]
                data = None
                if args.source in {"autotrader", "hybrid"}:
                    data = fetch_autotrader_prices(meta, args.max_pages, args.timeout, args.postcode)
                if (not data) and args.source in {"llm", "hybrid"} and client is not None:
                    data = call_llm(client, args.model, meta)
                return data

            futures = {ex.submit(runner, fid): fid for fid in to_run}
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                fid = futures[fut]
                data = None
                try:
                    data = fut.result()
                except Exception:
                    data = None
                cache[fid] = data or {}
                cf.write(json.dumps({"fid": fid, "data": cache[fid]}, ensure_ascii=False) + "\n")
                done += 1
                if done == 1 or done % 50 == 0:
                    rate = done / max(time.time() - start, 1e-6)
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"[price] {done}/{total} | {rate:.1f}/s | ETA ~{eta/60:.1f}m", flush=True)

    # Map back to rows
    lows: List[Optional[float]] = []
    highs: List[Optional[float]] = []
    confs: List[str] = []
    notes: List[str] = []
    for _, r in df.iterrows():
        dat = cache.get(r["__fid"], {})
        lo, hi, cf, nt = sanitize_prices(dat)
        # If still missing, keep existing value from file
        if lo is None:
            try:
                lo = float(str(r.get("Min. of used price low helper", "")).replace(",", ""))
            except Exception:
                lo = None
        if hi is None:
            try:
                hi = float(str(r.get("Max. of used price high helper", "")).replace(",", ""))
            except Exception:
                hi = None
        lows.append(lo)
        highs.append(hi)
        confs.append(cf)
        notes.append(nt)

    # Overwrite target numeric helper columns as requested
    df["Min. of used price low helper"] = lows
    df["Max. of used price high helper"] = highs
    df["price_confidence"] = confs
    df["price_notes"] = notes
    df.drop(columns=["__fid"], inplace=True)
    df.to_excel(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()


