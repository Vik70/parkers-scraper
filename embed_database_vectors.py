#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


def _load_api_key_from_env_file() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    if k.strip() == "OPENAI_API_KEY" and not os.getenv("OPENAI_API_KEY"):
                        os.environ["OPENAI_API_KEY"] = v.strip().strip('"').strip("'")
                        break
    except Exception:
        pass


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return vec
    return (arr / norm).tolist()


def _val(x) -> str:
    s = "" if x is None else str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def build_specs_string(row: pd.Series) -> str:
    parts: List[str] = []
    def add(label: str, value: str):
        if value:
            parts.append(f"{label}: {value}")

    # Core identifiers (lightweight)
    add("Body", _val(row.get("Real Body Type")))
    add("Engine", _val(row.get("engine type")))
    add("Power (bhp)", _val(row.get("Power (bhp)")))
    add("Transmission", _val(row.get("Transmission")))
    add("Doors", _val(row.get("Doors")))
    add("Seats", _val(row.get("Seats")))

    # Performance & economy (min/max where available)
    mpg_min = _val(row.get("Min. of mpg low helper"))
    mpg_max = _val(row.get("Max. of mpg high helper"))
    if mpg_min or mpg_max:
        if mpg_min and mpg_max and mpg_min != mpg_max:
            add("MPG", f"{mpg_min}-{mpg_max}")
        else:
            add("MPG", mpg_min or mpg_max)

    ig_min = _val(row.get("Min. of Insurance group"))
    ig_max = _val(row.get("Max. of Insurance group"))
    if ig_min or ig_max:
        if ig_min and ig_max and ig_min != ig_max:
            add("Insurance Group", f"{ig_min}-{ig_max}")
        else:
            add("Insurance Group", ig_min or ig_max)

    z_min = _val(row.get("Min. of 0-60 mph (secs)"))
    z_max = _val(row.get("Max. of 0-60 mph (secs)2"))
    if z_min or z_max:
        if z_min and z_max and z_min != z_max:
            add("0-60 mph (s)", f"{z_min}-{z_max}")
        else:
            add("0-60 mph (s)", z_min or z_max)

    # Dimensions
    add("Length (mm)", _val(row.get("Average of Length (mm)")))
    add("Width (mm)", _val(row.get("Average of Width (mm)")))
    add("Height (mm)", _val(row.get("Average of Height (mm)")))
    add("Luggage (L)", _val(row.get("Average of Luggage Capacity (litres)")))

    # Years
    ys = _val(row.get("Year start"))
    ye = _val(row.get("Year end"))
    if ys or ye:
        add("Years", f"{ys}-{ye}" if ys and ye else (ys or ye))

    return "; ".join(parts) if parts else ""


def build_specs_string_all(row: pd.Series, ignore_columns: Optional[set] = None, max_chars: int = 8000) -> str:
    ignore = ignore_columns or set()
    parts: List[str] = []
    total = 0
    for key, val in row.items():
        if key in ignore:
            continue
        s = _val(val)
        if not s:
            continue
        piece = f"{str(key)}: {s}"
        total += len(piece) + 2
        if total > max_chars:
            break
        parts.append(piece)
    out = "; ".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars]
    return out


def embed_batch(client, texts: List[str], model: str, max_retries: int = 5, backoff_base: float = 1.5) -> List[List[float]]:
    # Sanitize minimal: replace empty with 'placeholder' to avoid API errors
    safe_texts = [t if (t and t.strip()) else "placeholder" for t in texts]
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=model, input=safe_texts)
            return [d.embedding for d in resp.data]
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = backoff_base ** attempt
            print(f"[embed] transient error: {type(exc).__name__}; retry {attempt}/{max_retries} in {sleep_s:.1f}s", flush=True)
            time.sleep(sleep_s)


def chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {
        "Make": "make",
        "Model": "model",
        "Series (production years start-end)": "series_production_years",
        "Year start": "year_start",
        "Year end": "year_end",
        "Real Body Type": "real_body_type",
        "engine type": "engine_type",
        "Power (bhp)": "power_bhp",
        "Transmission": "transmission",
        "Doors": "doors",
        "Seats": "seats",
        "Min. of used price low helper": "used_price_low_min",
        "Max. of used price high helper": "used_price_high_max",
        "Min. of mpg low helper": "mpg_low_min",
        "Min. of Insurance group": "insurance_group_min",
        "Min. of 0-60 mph (secs)": "zero_to_sixty_secs_min",
        "Average of Length (mm)": "length_mm_avg",
        "Average of Width (mm)": "width_mm_avg",
        "Average of Height (mm)": "height_mm_avg",
        "Average of Luggage Capacity (litres)": "luggage_capacity_litres_avg",
        # passthroughs (keep as-is if present)
        "essence_main": "essence_main",
        "image_urls": "image_urls",
    }
    cols = {c: mapping.get(str(c), str(c)) for c in df.columns}
    return df.rename(columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed essence_main and compact specs into vector columns")
    ap.add_argument("--input", default="Database_1.0.xlsx")
    ap.add_argument("--output", default="Database_1.0_embedded.xlsx", help="Excel file for non-vector data (safe size)")
    ap.add_argument("--vectors-out", default="", help="Path to write vectors as JSONL (auto if empty)")
    ap.add_argument("--vectors-csv", default="", help="Optional CSV path (row_id,vector_main_json,vector_specs_json)")
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows (for smoke tests)")
    ap.add_argument("--output-csv", default="", help="Optional combined CSV of all columns + vectors (preserves headers unless --normalize-headers)")
    ap.add_argument("--normalize-headers", action="store_true", help="Normalize headers to snake_case for SQL")
    ap.add_argument("--sheet", default=None, help="Excel sheet name or index to read (e.g., Sheet2)")
    ap.add_argument("--specs-mode", choices=["compact", "all"], default="compact", help="Use compact selected fields or include all columns for specs text")
    ap.add_argument("--stream", action="store_true", help="Stream outputs chunk-by-chunk to CSV/JSONL to allow resume")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output files by skipping already written rows")
    args = ap.parse_args()

    read_kwargs: Dict[str, object] = {"dtype": str}
    if args.sheet is not None:
        read_kwargs["sheet_name"] = args.sheet
    df = pd.read_excel(args.input, **read_kwargs)
    if "essence_main" not in df.columns:
        raise SystemExit("Column 'essence_main' not found in input.")

    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()
        print(f"[info] Limiting to first {len(df)} rows", flush=True)

    # Build stable row_id for joining vectors (needed for streaming)
    def rid(row: pd.Series) -> str:
        return "|".join(
            [
                str(row.get("Make", "")).strip(),
                str(row.get("Model", "")).strip(),
                str(row.get("Series (production years start-end)", "")).strip(),
                str(row.get("Year start", "")).strip(),
                str(row.get("Year end", "")).strip(),
            ]
        )

    if "row_id" not in df.columns:
        df["row_id"] = df.apply(rid, axis=1)

    # Build specs strings
    if args.specs_mode == "all":
        ignore_cols = {"vector_main", "vector_specs", "row_id", "essence_main"}
        specs_strings = df.apply(lambda r: build_specs_string_all(r, ignore_cols), axis=1).astype(str).tolist()
    else:
        specs_strings = df.apply(build_specs_string, axis=1).astype(str).tolist()
    essence_texts = df["essence_main"].astype(str).fillna("").tolist()

    # Prepare client once
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise SystemExit("openai package not installed. Run: pip install openai") from exc

    _load_api_key_from_env_file()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI()

    total = len(df)
    print(f"[start] Embedding {total} rows | batch_size={args.batch_size} | model={args.model}", flush=True)

    # Streaming/resume mode writes outputs incrementally and skips already written rows
    if args.stream and (args.output_csv or args.vectors_out or args.vectors_csv):
        # Prepare base frame for CSV respecting header preference
        base_for_csv = normalize_headers(df.copy()) if args.normalize_headers else df.copy()
        # Determine resume starting index from output_csv if provided
        start_idx = 0
        if args.resume and args.output_csv and os.path.exists(args.output_csv):
            try:
                with open(args.output_csv, 'r', encoding='utf-8') as f:
                    # subtract header line
                    start_idx = max(sum(1 for _ in f) - 1, 0)
            except Exception:
                start_idx = 0
        if start_idx > 0:
            print(f"[resume] Skipping first {start_idx} rows based on existing CSV", flush=True)

        processed = 0
        start_time = time.time()
        i = start_idx
        # Ensure headers for CSV
        wrote_header_full = os.path.exists(args.output_csv) and start_idx > 0
        wrote_header_vec = os.path.exists(args.vectors_csv) and start_idx > 0
        # JSONL open mode
        jsonl_mode = 'a' if (args.vectors_out and os.path.exists(args.vectors_out) and start_idx > 0) else 'w'
        jsonl_f = open(args.vectors_out, jsonl_mode, encoding='utf-8') if args.vectors_out else None

        try:
            while i < total:
                j = min(i + max(1, args.batch_size), total)
                chunk_ess = essence_texts[i:j]
                chunk_spc = specs_strings[i:j]
                embs_main = [l2_normalize(e) for e in embed_batch(client, chunk_ess, args.model)]
                embs_specs = [l2_normalize(e) for e in embed_batch(client, chunk_spc, args.model)]

                # Build chunk frames and write
                if args.output_csv:
                    chunk_df = base_for_csv.iloc[i:j].copy()
                    chunk_df["vector_main"] = [json.dumps(v) for v in embs_main]
                    chunk_df["vector_specs"] = [json.dumps(v) for v in embs_specs]
                    chunk_df.to_csv(args.output_csv, mode='a', header=(not wrote_header_full), index=False)
                    wrote_header_full = True

                if args.vectors_csv:
                    chunk_vec_rows = []
                    for k, rid_val in enumerate(df["row_id"].iloc[i:j].tolist()):
                        chunk_vec_rows.append({
                            "row_id": rid_val,
                            "vector_main": json.dumps(embs_main[k]),
                            "vector_specs": json.dumps(embs_specs[k]),
                        })
                    pd.DataFrame(chunk_vec_rows).to_csv(args.vectors_csv, mode='a', header=(not wrote_header_vec), index=False)
                    wrote_header_vec = True

                if jsonl_f is not None:
                    for k, rid_val in enumerate(df["row_id"].iloc[i:j].tolist()):
                        rec = {"row_id": rid_val, "vector_main": embs_main[k], "vector_specs": embs_specs[k]}
                        jsonl_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    jsonl_f.flush()

                i = j
                processed = i - start_idx
                if processed == (j - start_idx) or processed % (args.batch_size * 5) == 0:
                    rate = processed / max(time.time() - start_time, 1e-6)
                    eta = ((total - start_idx) - processed) / rate if rate > 0 else 0
                    print(f"[stream] {processed}/{total - start_idx} (+{start_idx} skipped) | {rate:.1f}/s | ETA ~{eta/60:.1f}m", flush=True)
        finally:
            if jsonl_f is not None:
                jsonl_f.close()

        print("[done] Streaming outputs completed", flush=True)

        # Also write Excel copy without vectors for browsing
        df_without_vectors = df.drop(columns=["row_id"], errors="ignore").copy()
        df_without_vectors.to_excel(args.output, index=False)
        print(f"[done] Saved embedded workbook (no vectors in Excel): {args.output}")
        return

    # Non-streaming path (accumulate in memory, then write)
    vec_main: List[List[float]] = []
    vec_specs: List[List[float]] = []

    # Essence embeddings with progress
    chunks_main = chunked(essence_texts, max(1, args.batch_size))
    start = time.time()
    processed = 0
    for idx, chunk in enumerate(chunks_main, start=1):
        embs = embed_batch(client, chunk, args.model)
        vec_main.extend(l2_normalize(e) for e in embs)
        processed += len(chunk)
        if idx == 1 or idx % 5 == 0:
            rate = processed / max(time.time() - start, 1e-6)
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"[main] {processed}/{total} ({idx}/{len(chunks_main)}) | {rate:.1f}/s | ETA ~{eta/60:.1f}m", flush=True)

    # Specs embeddings with progress
    chunks_specs = chunked(specs_strings, max(1, args.batch_size))
    start2 = time.time()
    processed2 = 0
    for idx, chunk in enumerate(chunks_specs, start=1):
        embs = embed_batch(client, chunk, args.model)
        vec_specs.extend(l2_normalize(e) for e in embs)
        processed2 += len(chunk)
        if idx == 1 or idx % 5 == 0:
            rate = processed2 / max(time.time() - start2, 1e-6)
            eta = (total - processed2) / rate if rate > 0 else 0
            print(f"[specs] {processed2}/{total} ({idx}/{len(chunks_specs)}) | {rate:.1f}/s | ETA ~{eta/60:.1f}m", flush=True)

    if len(vec_main) != len(df) or len(vec_specs) != len(df):
        raise SystemExit("Embedding length mismatch with dataframe rows.")

    # Ensure row_id exists (non-stream path)
    if "row_id" not in df.columns:
        df["row_id"] = df.apply(rid, axis=1)

    # Write vectors to JSONL to avoid Excel's 32,767 char cell limit
    vectors_path = args.vectors_out or (
        os.path.splitext(args.output)[0] + "_vectors.jsonl"
    )
    with open(vectors_path, "w", encoding="utf-8") as f:
        for i in range(len(df)):
            rec = {
                "row_id": df.at[i, "row_id"],
                "vector_main": vec_main[i],
                "vector_specs": vec_specs[i],
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print(f"[done] Saved vectors JSONL: {vectors_path}")

    # Optional: also write CSV with JSON arrays as strings
    if args.vectors_csv:
        csv_rows = []
        for i in range(len(df)):
            csv_rows.append(
                {
                    "row_id": df.at[i, "row_id"],
                    "vector_main": json.dumps(vec_main[i]),
                    "vector_specs": json.dumps(vec_specs[i]),
                }
            )
        pd.DataFrame(csv_rows).to_csv(args.vectors_csv, index=False)
        print(f"[done] Saved vectors CSV: {args.vectors_csv}")

    # Optional: full CSV (all columns + vectors)
    if args.output_csv:
        full_df = normalize_headers(df.copy()) if args.normalize_headers else df.copy()
        full_df["vector_main"] = [json.dumps(v) for v in vec_main]
        full_df["vector_specs"] = [json.dumps(v) for v in vec_specs]
        full_df.to_csv(args.output_csv, index=False)
        print(f"[done] Saved full CSV with vectors: {args.output_csv}")

    # Save non-vector data to Excel (safer for human viewing)
    df_without_vectors = df.drop(columns=["row_id"], errors="ignore").copy()
    df_without_vectors.to_excel(args.output, index=False)
    print(f"[done] Saved embedded workbook (no vectors in Excel): {args.output}")


if __name__ == "__main__":
    main()


