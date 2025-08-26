#!/usr/bin/env python3
import argparse
import sys
import re
from pathlib import Path
import pandas as pd

try:
    from clean_specs_boilerplate import is_specs_boilerplate
except Exception:
    def is_specs_boilerplate(text: str) -> bool:
        if not isinstance(text, str) or not text:
            return False
        t = text.lower()
        strong_pattern = r"follow\s+(?:three|3)\s+simple\s+steps\s+to\s+get\s+to\s+your\s+desired\s+version"
        patterns = [
            strong_pattern,
            r"step\s*1\s*of\s*3\s*-\s*select\s*trim\s*level",
            r"step\s*2\s*of\s*3\s*-\s*select\s*engine",
            r"step\s*3\s*of\s*3\s*-\s*select\s*version",
            r"select\s+one\s+of\s+the\s+versions\s+below",
            r"brief\s+specification\s+overview",
            r"enter\s+the\s+car\s+registration",
        ]
        if re.search(strong_pattern, t, flags=re.IGNORECASE):
            return True
        matches = sum(1 for p in patterns if re.search(p, t, flags=re.IGNORECASE))
        return matches >= 2


def normalize_key(value) -> str:
    s = "" if value is None else str(value)
    s = re.sub(r"\s+", " ", s.strip()).lower()
    return s


def pick_best_string(values) -> str:
    strings = [v for v in values if isinstance(v, str)]
    non_empty = [s for s in strings if s.strip()]
    non_boiler = [s for s in non_empty if not is_specs_boilerplate(s)]
    candidates = non_boiler if non_boiler else non_empty
    if not candidates:
        return ""
    # Prefer the longest non-boilerplate text (tends to be the full article text)
    return max(candidates, key=lambda s: len(s))


def _find_year_column(df: pd.DataFrame, preferred: list[str] | None = None) -> str | None:
    preferred = preferred or []
    # 1) Exact preferred names
    for name in preferred:
        if name in df.columns:
            return name
    # 2) Common exact names
    for name in [
        "Series_Production_Year",
        "Series prod year",
        "Series production year",
        "Series Production Year",
        "Series (production years start-end)",
    ]:
        if name in df.columns:
            return name
    # 3) Fuzzy match: require tokens 'series' & ('prod' or 'production') & 'year'
    def normalize(n: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", n.lower()).strip()
    for col in df.columns:
        norm = normalize(col)
        tokens = set(norm.split())
        has_series = "series" in tokens
        has_year = ("year" in tokens) or ("years" in tokens)
        has_prod = ("prod" in tokens) or ("production" in tokens)
        if has_series and has_year and has_prod:
            return col
    return None


def build_aggregated_lookup(small_df: pd.DataFrame) -> pd.DataFrame:
    # Identify possible year column in the small file
    year_col = _find_year_column(small_df)
    if not year_col:
        raise ValueError("Could not locate a 'series production year' column in small file")

    # Ensure target columns exist
    for col in ["Article_Text", "Pros", "Cons", "Rivals"]:
        if col not in small_df.columns:
            small_df[col] = ""

    # Create normalized key
    small_df = small_df.copy()
    small_df["__year_key__"] = small_df[year_col].apply(normalize_key)

    # Aggregate per year key by choosing the best string per field
    grouped = small_df.groupby("__year_key__", dropna=False).agg({
        "Article_Text": pick_best_string,
        "Pros": pick_best_string,
        "Cons": pick_best_string,
        "Rivals": pick_best_string,
    }).reset_index()

    return grouped.rename(columns={"__year_key__": "__join_key__"})


def merge_into_big(big_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    # Determine big year column via fuzzy matching
    big_year_col = _find_year_column(big_df)
    if not big_year_col:
        raise ValueError("Could not locate a 'series production year' column in big file")

    big_df = big_df.copy()
    big_df["__join_key__"] = big_df[big_year_col].apply(normalize_key)

    merged = big_df.merge(lookup_df, on="__join_key__", how="left")

    # If these columns exist in big, overwrite; else create new columns
    for col in ["Article_Text", "Pros", "Cons", "Rivals"]:
        if col in big_df.columns:
            merged[col] = merged[col + "_y"].combine_first(merged[col + "_x"]) if (col + "_y") in merged.columns else merged[col]
        else:
            # The aggregated columns are already named correctly in lookup; ensure present
            if col not in merged.columns:
                merged[col] = ""

    # Drop helper columns and any merge suffix columns
    drop_cols = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y") or c == "__join_key__"]
    merged = merged.drop(columns=drop_cols, errors='ignore')

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge 2.7k descriptions (pros/cons/article_text/rivals) into 13k pivot file by series production year.")
    parser.add_argument("--small", default=None, help="Path to small file (default: autodetect cleaned/enhanced)")
    parser.add_argument("--big", default="Tabulated Pivots ONLY .xlsx", help="Path to 13k pivot Excel")
    parser.add_argument("--output", default=None, help="Output Excel path (default: '<big>_enriched.xlsx')")
    args = parser.parse_args()

    # Resolve small file path
    small_candidates = [
        args.small,
        "Series + URL_sequential_enhanced_cleaned.xlsx",
        "Series + URL_sequential_enhanced.xlsx",
        "Series + URL_enhanced.xlsx",
    ]
    small_path = None
    for cand in small_candidates:
        if cand and Path(cand).exists():
            small_path = Path(cand)
            break
    if not small_path:
        print("❌ Could not find small source Excel. Provide with --small.")
        sys.exit(1)

    big_path = Path(args.big)
    if not big_path.exists():
        print(f"❌ Big pivot Excel not found: {big_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else big_path.with_name(big_path.stem + "_enriched.xlsx")

    print(f"📄 Loading small source: {small_path}")
    small_df = pd.read_excel(small_path)

    print(f"📄 Loading big pivot: {big_path}")
    big_df = pd.read_excel(big_path)

    print("🔧 Building aggregated lookup by series production year…")
    lookup_df = build_aggregated_lookup(small_df)
    print(f"📊 Lookup keys: {lookup_df.shape[0]}")

    print("🔗 Merging into big pivot…")
    merged_df = merge_into_big(big_df, lookup_df)

    # Basic stats
    matched = merged_df[["Article_Text", "Pros", "Cons", "Rivals"]].notna().any(axis=1).sum()
    print(f"✅ Rows with at least one field added: {matched} / {len(merged_df)}")

    print(f"💾 Saving to: {output_path}")
    merged_df.to_excel(output_path, index=False)
    print("🎉 Done.")


if __name__ == "__main__":
    main()


