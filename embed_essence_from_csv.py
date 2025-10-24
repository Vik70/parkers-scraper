#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from typing import List


def load_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    env = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env):
        for line in open(env, "r", encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.split("=", 1)[0].strip() == "OPENAI_API_KEY":
                os.environ["OPENAI_API_KEY"] = s.split("=", 1)[1].strip().strip('"').strip("'")
                break


def chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed essence_main from a minimal CSV and output vectors CSV")
    ap.add_argument("--input", default="db2_sheet2_min.csv")
    ap.add_argument("--out", default="db2_main_vectors.csv")
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    texts = [row.get("essence_main", "") or "" for row in rows]

    from openai import OpenAI
    load_api_key()
    client = OpenAI()

    vectors: List[List[float]] = []
    total = len(texts)
    start = time.time()
    done = 0
    for idx, ch in enumerate(chunked(texts, max(1, args.batch_size)), start=1):
        safe = [t if t.strip() else "placeholder" for t in ch]
        emb = client.embeddings.create(model=args.model, input=safe)
        for d in emb.data:
            vectors.append(d.embedding)
        done += len(ch)
        if idx == 1 or idx % 5 == 0:
            rate = done / max(time.time() - start, 1e-6)
            eta = (total - done) / rate if rate > 0 else 0
            print(f"[embed] {done}/{total} | {rate:.1f}/s | ETA ~{eta/60:.1f}m", flush=True)

    # write output CSV with bracketed arrays
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Make","Model","Series (production years start-end)","Year start","Year end","vector_main_text"])
        for row, vec in zip(rows, vectors):
            w.writerow([
                row.get("Make",""),
                row.get("Model",""),
                row.get("Series (production years start-end)",""),
                row.get("Year start",""),
                row.get("Year end",""),
                json.dumps(vec),
            ])
    print(f"[done] Wrote vectors: {args.out}")


if __name__ == "__main__":
    main()



