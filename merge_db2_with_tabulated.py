#!/usr/bin/env python3
import argparse
import os
import pandas as pd


SERIES_COL = "Series (production years start-end)"


def normalize_years(value: str) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    for dash in ("‐", "‑", "‒", "–", "—", "―", "−", "to"):
        s = s.replace(dash, "-")
    s = s.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge essence_main + Image URL into database_2.0 by matching series years")
    ap.add_argument("--db2", default="database_2.0.xlsx")
    ap.add_argument("--tab", default="Tabulated Pivots ONLY_enhanced_llm_fresh.xlsx")
    ap.add_argument("--out", default="database_2.0_enriched.xlsx")
    args = ap.parse_args()

    db2 = pd.read_excel(args.db2, dtype=str)
    tab = pd.read_excel(args.tab, dtype=str)

    if SERIES_COL not in db2.columns or SERIES_COL not in tab.columns:
        raise SystemExit(f"Missing series column in inputs: {SERIES_COL}")

    # Build mapping from tabulated
    t = tab[[SERIES_COL, "essence_main", "Image URL"]].copy()
    t[SERIES_COL] = t[SERIES_COL].map(normalize_years)
    # For duplicates, keep first non-empty essence_main/Image URL
    t.sort_values(by=[SERIES_COL], inplace=True)
    t = t.groupby(SERIES_COL, as_index=False).agg({
        "essence_main": lambda s: next((x for x in s.tolist() if str(x).strip()), ""),
        "Image URL": lambda s: next((x for x in s.tolist() if str(x).strip()), ""),
    })

    d = db2.copy()
    d[SERIES_COL] = d[SERIES_COL].map(normalize_years)
    out = d.merge(t, on=SERIES_COL, how="left")
    # Ensure columns exist even if no match
    if "essence_main" not in out.columns:
        out["essence_main"] = ""
    if "Image URL" not in out.columns:
        out["Image URL"] = ""

    out.to_excel(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()




