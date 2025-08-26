#!/usr/bin/env python3
import re
import sys
import argparse
import pandas as pd
from pathlib import Path


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

    # If the strong pattern appears, consider it boilerplate immediately
    if re.search(strong_pattern, t, flags=re.IGNORECASE):
        return True

    # Otherwise, require at least 2 weaker signals to avoid false positives
    matches = sum(1 for p in patterns if re.search(p, t, flags=re.IGNORECASE))
    return matches >= 2


def clean_column(df: pd.DataFrame, column_name: str) -> int:
    if column_name not in df.columns:
        return 0
    mask = df[column_name].apply(is_specs_boilerplate)
    affected = int(mask.sum())
    if affected:
        df.loc[mask, column_name] = ""
    return affected


def main():
    parser = argparse.ArgumentParser(description="Remove specs boilerplate descriptions from Excel.")
    parser.add_argument("--input", required=True, help="Path to input Excel (e.g., 'Series + URL_sequential_enhanced.xlsx')")
    parser.add_argument("--output", help="Optional output path; defaults to '<input>_cleaned.xlsx'")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_cleaned.xlsx")

    print(f"📄 Loading: {input_path}")
    df = pd.read_excel(input_path)

    total_affected = 0
    # Prefer 'Article_Text'; also handle 'Description' if present
    for col in ["Article_Text", "Description"]:
        affected = clean_column(df, col)
        if affected:
            print(f"🧹 Cleared {affected} boilerplate descriptions in column '{col}'")
            total_affected += affected

    # If neither column existed, inform and exit gracefully
    if total_affected == 0 and not any(c in df.columns for c in ["Article_Text", "Description"]):
        print("⚠️  Neither 'Article_Text' nor 'Description' column found; no changes made.")
    else:
        print(f"✅ Total rows cleaned: {total_affected}")

    print(f"💾 Saving cleaned file to: {output_path}")
    df.to_excel(output_path, index=False)
    print("🎉 Done.")


if __name__ == "__main__":
    main()


