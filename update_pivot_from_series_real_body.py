#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Optional, Tuple

import pandas as pd


SERIES_COL = "Series (production years start-end)"
REAL_BODY_COL = "Real Body Type"


def normalize_years(value: str) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.strip().lower()
    if not s:
        return ""
    # unify dash variants
    for dash in ["‐", "‑", "‒", "–", "—", "―", "−", "to", "–", "—"]:
        s = s.replace(dash, "-")
    # collapse spaces around dashes
    s = s.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    # collapse interior multiple spaces
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def load_series_mapping(path: str) -> Dict[str, str]:
    df = pd.read_excel(path, dtype=str)
    if SERIES_COL not in df.columns or REAL_BODY_COL not in df.columns:
        raise SystemExit(f"Expected columns not found in series file: {SERIES_COL!r}, {REAL_BODY_COL!r}")
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        key = normalize_years(row.get(SERIES_COL, ""))
        val = (row.get(REAL_BODY_COL, "") or "").strip()
        if key and val and key not in mapping:
            mapping[key] = val
    return mapping


def find_header_row_no_header(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    # Inspect first 200 rows for a header row containing 'series' and 'year'
    max_scan = min(200, len(df))
    candidate_idx: Optional[int] = None
    for i in range(max_scan):
        vals = [str(v).strip() for v in df.iloc[i].tolist()]
        joined = "\t".join(v.lower() for v in vals if v and v != "nan")
        if not joined:
            continue
        if ("series" in joined and "year" in joined) or SERIES_COL.lower() in joined:
            candidate_idx = i
            break
    if candidate_idx is None:
        # fallback: first non-empty row
        for i in range(max_scan):
            vals = [str(v).strip() for v in df.iloc[i].tolist()]
            if any(v for v in vals if v and v != "nan"):
                candidate_idx = i
                break
    if candidate_idx is None:
        candidate_idx = 0
    header = [str(v).strip() if str(v).strip() != "nan" else "" for v in df.iloc[candidate_idx].tolist()]
    # Make unique column names
    seen = {}
    cols = []
    for j, h in enumerate(header):
        name = h or f"Unnamed: {j}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    data = df.iloc[candidate_idx + 1 :].copy()
    data.columns = cols
    # Drop columns that are entirely empty
    data = data.dropna(axis=1, how="all")
    return candidate_idx, data


def choose_series_column(cols: pd.Index) -> Optional[str]:
    if SERIES_COL in cols:
        return SERIES_COL
    low = [c.lower() for c in cols]
    # Try a heuristic: contains 'series' and 'year'
    for c, lc in zip(cols, low):
        if "series" in lc and "year" in lc:
            return c
    # Fallback: any column containing 'series'
    for c, lc in zip(cols, low):
        if "series" in lc:
            return c
    return None


def update_pivot(pivot_path: str, series_path: str, output_path: str, sheet: Optional[str] = None) -> None:
    mapping = load_series_mapping(series_path)
    # Load pivot without headers, then detect header row
    raw = pd.read_excel(pivot_path, sheet_name=sheet, header=None)
    _, p = find_header_row_no_header(raw)
    series_col = choose_series_column(p.columns)
    if not series_col:
        raise SystemExit("Could not find a series years column in the pivot file.")
    p["__series_key"] = p[series_col].astype(str).map(normalize_years)
    p[REAL_BODY_COL] = p["__series_key"].map(mapping).fillna("")
    p.drop(columns=["__series_key"], inplace=True)
    p.to_excel(output_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Update pivot body types by matching series years to Real Body Type mapping")
    ap.add_argument("--pivot", default="UPDATED PIVOT 01.09.xlsx")
    ap.add_argument("--series", default="Series + URL_with_real_body_type.xlsx")
    ap.add_argument("--output", default="UPDATED PIVOT 01.09_with_real_body_type.xlsx")
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.pivot):
        raise SystemExit(f"Pivot file not found: {args.pivot}")
    if not os.path.exists(args.series):
        raise SystemExit(f"Series mapping file not found: {args.series}")
    update_pivot(args.pivot, args.series, args.output, args.sheet)
    print(f"Saved updated pivot with '{REAL_BODY_COL}': {args.output}")


if __name__ == "__main__":
    main()


