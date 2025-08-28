import argparse
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Export embedded Excel to ordered CSV for Supabase import")
    ap.add_argument("--input", required=True, help="Embedded Excel path (with vector_* columns)")
    ap.add_argument("--output", required=True, help="CSV output path")
    ap.add_argument("--series-col", default="Series (production years start-end)", help="Original series years column name in Excel")
    args = ap.parse_args()

    df = pd.read_excel(args.input, dtype=str)

    # Rename Excel column to db-friendly name
    if args.series_col in df.columns:
        df = df.rename(columns={args.series_col: "series_years"})

    # Ensure vector columns are present and rename to *_text for staging import
    rename_map = {
        "vector_main": "vector_main_text",
        "vector_specs": "vector_specs_text",
        "vector_rivals": "vector_rivals_text",
    }
    for old, new in rename_map.items():
        if old not in df.columns:
            raise SystemExit(f"Missing expected column: {old}")
    df = df.rename(columns=rename_map)

    keep = [
        "Make",
        "Model",
        "series_years",
        "essence_main",
        "specs_summary_compact",
        "rivals_json_family",
        "vector_main_text",
        "vector_specs_text",
        "vector_rivals_text",
    ]
    for k in keep:
        if k not in df.columns:
            raise SystemExit(f"Missing expected column for export: {k}")

    out = df[keep].copy()
    out.columns = [
        "make",
        "model",
        "series_years",
        "essence_main",
        "specs_summary_compact",
        "rivals_json_family",
        "vector_main_text",
        "vector_specs_text",
        "vector_rivals_text",
    ]

    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {args.output} with shape {out.shape}")


if __name__ == "__main__":
    main()


