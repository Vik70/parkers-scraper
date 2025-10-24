#!/usr/bin/env python3
import argparse
import csv
import os
import pandas as pd


def build_row_id(row: pd.Series) -> str:
    return "|".join(
        [
            str(row.get("Make", "")).strip(),
            str(row.get("Model", "")).strip(),
            str(row.get("Series (production years start-end)", "")).strip(),
            str(row.get("Year start", "")).strip(),
            str(row.get("Year end", "")).strip(),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build safe staging CSV for Supabase/pgvector (vectors as text)")
    ap.add_argument("--excel", default="database_2.0_enriched.xlsx")
    ap.add_argument("--sheet", default="Sheet2")
    ap.add_argument("--vectors_csv", default="database_2.0_vectors.csv")
    ap.add_argument("--out", default="database_2.0_full_for_pgvector.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.excel, sheet_name=args.sheet, dtype=str)
    if "essence_main" not in df.columns:
        raise SystemExit("essence_main missing from Excel sheet")

    if "row_id" not in df.columns:
        df["row_id"] = df.apply(build_row_id, axis=1)

    vec = pd.read_csv(args.vectors_csv, dtype=str)
    if not set(["row_id", "vector_main", "vector_specs"]).issubset(set(vec.columns)):
        raise SystemExit("vectors CSV must contain row_id, vector_main, vector_specs")

    merged = df.merge(vec, on="row_id", how="left")
    # rename vector cols to text fields for staging
    merged = merged.rename(columns={"vector_main": "vector_main_text", "vector_specs": "vector_specs_text"})

    # Write CSV with strict quoting to avoid importer parse issues
    merged.to_csv(args.out, index=False, quoting=csv.QUOTE_ALL, line_terminator="\n", encoding="utf-8")
    print(f"Wrote: {args.out} ({len(merged)} rows, {len(merged.columns)} cols)")


if __name__ == "__main__":
    main()



