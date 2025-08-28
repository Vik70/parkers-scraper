import argparse
import pandas as pd
from typing import Dict


SOURCE_TO_TARGET: Dict[str, str] = {
    "engine type": "engine_type",
    "Transmission": "transmission_type",
    "Min. of Power (bhp)": "bhp_min",
    "Max. of Power (bhp)": "bhp_max",
    "Min. of Insurance group": "insurance_group_min",
    "Max. of Insurance group": "insurance_group_max",
    "Min. of mpg low helper": "mpg_min",
    "Max. of mpg high helper": "mpg_max",
    "Min. of used price low helper": "price_min",
    "Max. of used price high helper": "price_max",
}


def coerce_int_series(s: pd.Series) -> pd.Series:
    # Coerce to integer where possible; leave empty string otherwise
    out = pd.to_numeric(s, errors="coerce")
    return out.round().astype("Int64").astype(str).replace({"<NA>": ""})


def main() -> None:
    ap = argparse.ArgumentParser(description="Export structured filter fields from XLSX to CSV with exact target column names")
    ap.add_argument("--input", required=True, help="Path to Excel source (e.g., 'Tabulated Pivots ONLY_enriched_with_essence_v2_texts_embedded.xlsx')")
    ap.add_argument("--output", required=True, help="CSV output path")
    ap.add_argument("--make-col", default="Make")
    ap.add_argument("--model-col", default="Model")
    ap.add_argument("--years-col", default="Series (production years start-end)")
    args = ap.parse_args()

    df = pd.read_excel(args.input, dtype=str)

    cols_out = []
    data = {}

    # Always include keys to allow joins/updates later
    for src, tgt in [(args.make_col, "make"), (args.model_col, "model"), (args.years_col, "series_years")]:
        if src in df.columns:
            data[tgt] = df[src].astype(str)
            cols_out.append(tgt)

    # Map specified structured fields
    for src, tgt in SOURCE_TO_TARGET.items():
        if src not in df.columns:
            # Create empty column if missing
            data[tgt] = [""] * len(df)
            cols_out.append(tgt)
            continue
        series = df[src].astype(str)
        if tgt.endswith(("_min", "_max")) and tgt not in ("engine_type", "transmission_type"):
            series = coerce_int_series(series)
        data[tgt] = series
        cols_out.append(tgt)

    out_df = pd.DataFrame(data, columns=cols_out)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {args.output} with shape {out_df.shape}")


if __name__ == "__main__":
    main()


