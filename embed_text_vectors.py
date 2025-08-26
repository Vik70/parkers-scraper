#!/usr/bin/env python3
"""
Embeds three text columns into vector columns with L2-normalization:

- vector_main            ← embed essence_main (or essence_family fallback)
- vector_specs           ← embed specs_summary_compact (facts only)
- vector_rivals          ← embed compact rivals string from rivals_json_family

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
        dotenv_path = Path(".env")
        if dotenv_path.exists():
            try:
                for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "=" in s:
                        k, v = s.split("=", 1)
                        if k.strip() == "OPENAI_API_KEY" and not os.getenv("OPENAI_API_KEY"):
                            os.environ["OPENAI_API_KEY"] = v.strip().strip('"').strip("'")
                            api_key = os.environ["OPENAI_API_KEY"]
                            break
            except Exception:
                pass
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in environment or .env")

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_embedded.xlsx")

    df = pd.read_excel(in_path)

    # Choose main text: prefer essence_main; fallback to essence_family
    if "essence_main" in df.columns:
        main_series = df["essence_main"].astype(str).fillna("")
    elif "essence_family" in df.columns:
        main_series = df["essence_family"].astype(str).fillna("")
    else:
        raise SystemExit("Missing required text column: essence_main or essence_family")

    need_cols = ["rivals_json_family"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    # Prepare input strings
    fam_texts = main_series.tolist()
    riv_texts = [rivals_to_string(v) for v in df["rivals_json_family"].tolist()]
    specs_texts = df.get("specs_summary_compact", pd.Series([""] * len(df))).astype(str).fillna("").tolist()

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
    spec_vecs = embed_series(specs_texts)

    df["vector_main"] = [json.dumps(v) for v in fam_vecs]
    df["vector_rivals"] = [json.dumps(v) for v in riv_vecs]
    df["vector_specs"] = [json.dumps(v) for v in spec_vecs]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


