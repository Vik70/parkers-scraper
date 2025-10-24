#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd


def summarize_column(df: pd.DataFrame, col: str, excel_char_limit: int) -> None:
    if col not in df.columns:
        print(f"[warn] Column not found: {col}")
        return
    vals = df[col].astype(str).fillna("").tolist()
    n = len(vals)
    parse_fail = 0
    length_counts: Counter = Counter()
    max_len = 0
    flagged_len = 0
    bad_examples: List[Tuple[int, int]] = []  # (row_idx, str_len)
    dims: Counter = Counter()
    norm_issues = 0

    for i, s in enumerate(vals):
        L = len(s)
        max_len = max(max_len, L)
        if L >= excel_char_limit:
            flagged_len += 1
        try:
            vec = json.loads(s)
            if not isinstance(vec, list) or not vec:
                parse_fail += 1
                if len(bad_examples) < 5:
                    bad_examples.append((i, L))
                continue
            dims[len(vec)] += 1
            # light norm check on a small sample
            if i < 10:
                arr = np.asarray(vec, dtype=float)
                norm = float(np.linalg.norm(arr))
                if norm < 0.97 or norm > 1.03:
                    norm_issues += 1
        except Exception:
            parse_fail += 1
            if len(bad_examples) < 5:
                bad_examples.append((i, L))

    most_common_dim = dims.most_common(1)[0][0] if dims else 0
    unique_dims = sorted(dims.keys())
    print(f"[col={col}] rows={n} | parse_fail={parse_fail} | max_char_len={max_len} | >=limit={flagged_len}")
    print(f"[col={col}] dims (top): {dims.most_common(3)} | unique_dims={unique_dims[:5]}{'...' if len(unique_dims)>5 else ''}")
    if bad_examples:
        print(f"[col={col}] examples of failures/odd rows (idx, strlen): {bad_examples}")
    if norm_issues:
        print(f"[col={col}] norm deviations in first 10 rows: {norm_issues}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check embedded vector columns for truncation and consistency")
    ap.add_argument("--input", required=True)
    ap.add_argument("--vec-cols", nargs="*", default=["vector_main", "vector_specs"]) 
    ap.add_argument("--excel-char-limit", type=int, default=32767)
    args = ap.parse_args()

    df = pd.read_excel(args.input, dtype=str)
    for col in args.vec_cols:
        summarize_column(df, col, args.excel_char_limit)


if __name__ == "__main__":
    main()


