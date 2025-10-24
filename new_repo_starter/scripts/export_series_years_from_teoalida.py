import argparse
import re
from typing import Optional

import pandas as pd


def col_letter_to_index(letter: str) -> int:
    s = letter.strip().upper()
    if not re.fullmatch(r"[A-Z]+", s):
        raise ValueError(f"Invalid column letter: {letter}")
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - ord('A') + 1)
    return n  # 1-based


def pick_series_col(df: pd.DataFrame, series_col_name: Optional[str], series_col_letter: Optional[str]) -> pd.Series:
    if series_col_name and series_col_name in df.columns:
        return df[series_col_name]
    if series_col_letter:
        idx1 = col_letter_to_index(series_col_letter)
        return df.iloc[:, idx1 - 1]
    # fallback: try common header names
    for cand in (
        "Series Production Years",
        "Series prod years",
        "Series years",
        "Production years",
        "Series (production years start-end)",
        "Series (production years)",
    ):
        if cand in df.columns:
            return df[cand]
    raise SystemExit("Could not locate series production years column. Pass --series-col-name or --series-col-letter (e.g., BN).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract make/model/series_years from Teoalida XLSX (Version Table sheet)")
    ap.add_argument("--input", required=True, help="United-Kingdom-Car-Database-by-Teoalida-full-specs.xlsx")
    ap.add_argument("--output", required=True, help="vehicles_series_years.csv")
    ap.add_argument("--sheet", default="Version Table")
    ap.add_argument("--make-col", default="Make")
    ap.add_argument("--model-col", default="Model")
    ap.add_argument("--series-col-name", default=None, help="Header name for series col if available")
    ap.add_argument("--series-col-letter", default="BN", help="Excel column letter when header unknown (e.g., BN)")
    args = ap.parse_args()

    # Read with no header, find header row dynamically (many Teoalida sheets have multi-row headers)
    # Read without dtype forcing, then coerce to string safely for header detection
    raw = pd.read_excel(args.input, sheet_name=args.sheet, header=None)
    norm = raw.fillna("").astype(str).applymap(lambda x: x.strip().lower())
    hdr_idx = None
    # Prefer row containing all three, else Make/Model
    for i in range(min(len(raw), 200)):
        vals = set(norm.iloc[i].tolist())
        if all(k in vals for k in [args.make_col.lower(), args.model_col.lower()]):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise SystemExit("Could not detect header row containing Make/Model in sheet; please verify sheet name.")

    df = pd.read_excel(args.input, sheet_name=args.sheet, header=hdr_idx, dtype=str)
    # strip header whitespace
    df.rename(columns={c: (c.strip() if isinstance(c, str) else c) for c in df.columns}, inplace=True)
    if args.make_col not in df.columns or args.model_col not in df.columns:
        raise SystemExit(f"Missing required columns after header detection: {args.make_col} and/or {args.model_col}")

    make = df[args.make_col].astype(str).fillna("")
    model = df[args.model_col].astype(str).fillna("")
    series = pick_series_col(df, args.series_col_name, args.series_col_letter).astype(str).fillna("")

    out = pd.DataFrame({
        "make": make.str.strip(),
        "model": model.str.strip(),
        "series_years": series.str.strip(),
    })

    # Drop completely empty keys
    out = out[(out["make"] != "") & (out["model"] != "")]
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {args.output} with {len(out)} rows")


if __name__ == "__main__":
    main()


