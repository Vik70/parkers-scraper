#!/usr/bin/env python3
"""
Embeds three text columns into vector columns with L2-normalization:

- vector_family_essence  ← embed essence_family
- vector_family_rivals   ← embed compact rivals string from rivals_json_family
- vector_variant_delta   ← embed essence_variant_delta

Usage:
  set OPENAI_API_KEY=...  (Windows PowerShell: $env:OPENAI_API_KEY="...")
  python embed_text_vectors.py --input <xlsx> --output <xlsx>

Notes:
- Uses text-embedding-3-small by default
- Stores vectors as JSON arrays in Excel cells
"""

import argparse
import ast
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise SystemExit("openai package not installed. Run: pip install openai") from exc


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return vec
    return (arr / norm).tolist()


def rivals_to_string(val: str) -> str:
    try:
        data = ast.literal_eval(val) if isinstance(val, str) else (val or [])
    except Exception:
        data = []
    parts = []
    for item in (data or []):
        name = (item.get("name") or "").strip()
        ctx = (item.get("context") or "").strip()
        if name:
            if ctx:
                parts.append(f"{name}: {ctx}")
            else:
                parts.append(name)
    return " | ".join(parts)


def embed_texts(texts: List[str], client: OpenAI, model: str) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=model, input=texts)
    # openai>=1.0 returns data list with .embedding
    return [d.embedding for d in resp.data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed essence/rivals/delta texts to vector columns")
    parser.add_argument("--input", required=False, default="Tabulated Pivots ONLY_enriched_with_essence_v2_texts.xlsx")
    parser.add_argument("--output", required=False, default=None)
    parser.add_argument("--model", required=False, default="text-embedding-3-small")
    parser.add_argument("--batch", type=int, default=256, help="Batch size for embedding calls")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in environment")

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_embedded.xlsx")

    df = pd.read_excel(in_path)

    need_cols = ["essence_family", "rivals_json_family", "essence_variant_delta"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    # Prepare input strings
    fam_texts = df["essence_family"].astype(str).fillna("").tolist()
    riv_texts = [rivals_to_string(v) for v in df["rivals_json_family"].tolist()]
    var_texts = df["essence_variant_delta"].astype(str).fillna("").tolist()

    client = OpenAI()

    def embed_series(texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), args.batch):
            chunk = texts[i : i + args.batch]
            raw_vecs = embed_texts(chunk, client, args.model)
            vectors.extend([l2_normalize(v) for v in raw_vecs])
        return vectors

    fam_vecs = embed_series(fam_texts)
    riv_vecs = embed_series(riv_texts)
    var_vecs = embed_series(var_texts)

    df["vector_family_essence"] = [json.dumps(v) for v in fam_vecs]
    df["vector_family_rivals"] = [json.dumps(v) for v in riv_vecs]
    df["vector_variant_delta"] = [json.dumps(v) for v in var_vecs]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


