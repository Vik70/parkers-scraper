#!/usr/bin/env python3
import argparse
import json
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate CSV vector columns for truncation and dims")
    ap.add_argument("--input", default="database_2.0_full.csv")
    ap.add_argument("--cols", nargs="*", default=["vector_main", "vector_specs"])
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"rows={len(df)} cols={len(df.columns)}")
    for col in args.cols:
        if col not in df.columns:
            print(f"[warn] missing column: {col}")
            continue
        errs = 0
        dims = set()
        max_len = 0
        for s in df[col].astype(str):
            if len(s) > max_len:
                max_len = len(s)
            try:
                v = json.loads(s)
                dims.add(len(v))
            except Exception:
                errs += 1
        print(f"{col}: dims={sorted(dims)[:5]} errs={errs} max_str_len={max_len}")


if __name__ == "__main__":
    main()




