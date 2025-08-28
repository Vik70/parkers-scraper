import json
import math
import os
import sys

import pandas as pd


def main(path: str) -> None:
    if not os.path.exists(path):
        print(f"missing: {path}")
        sys.exit(1)
    df = pd.read_excel(path, dtype=str)
    n = len(df)
    cols = ["vector_main", "vector_specs", "vector_rivals"]
    print("rows:", n)
    for c in cols:
        print(c, "present" if c in df.columns else "MISSING")
    # counts
    for c in cols:
        if c in df.columns:
            cnt = df[c].astype(str).str.startswith("[").sum()
            print(c, "vector-like cells:", cnt)
    # sample norms
    def l2(v):
        return math.sqrt(sum(float(x) * float(x) for x in v))
    sample_idxs = [0, 1, 2, 100, 500, 1000]
    sample_idxs = [i for i in sample_idxs if i < n]
    for c in cols:
        if c not in df.columns:
            continue
        norms = []
        for i in sample_idxs:
            try:
                arr = json.loads(df.at[i, c])
                if isinstance(arr, list) and arr:
                    norms.append(round(l2(arr), 4))
                else:
                    norms.append(None)
            except Exception:
                norms.append(None)
        print(c, "sample_norms:", norms)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "tmp_check_embedded.xlsx"
    main(p)


