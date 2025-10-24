#!/usr/bin/env python3
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

import pandas as pd


PROMPT_TEXT = (
    "Label each car with 2–3 words in the format:\n"
    "[Base] [Descriptor] [Descriptor?]\n\n"
    "Rules\n"
    "- First word MUST be the Base.\n"
    "- Base (choose exactly one): SUV, Estate, Coupe, Sports Car, Supercar, Saloon, Hatchback,\n"
    "  Convertible, Van, Gran Coupe, GT\n"
    "- Add 1–2 short UK-style descriptors for size/shape/positioning (e.g., Compact, Midsize,\n"
    "  Large, Executive, Luxury, Sportback, Coupe Styled, Off-Road, Performance, Premium).\n"
    "- If the Base itself is two words (e.g., “Sports Car”), use at most ONE descriptor (keep total ≤ 3 words).\n"
    "- If the Base is Supercar DO NOT add any other words (total = 1 word)\n"
    "- Do NOT repeat the base in the descriptors (avoid “Coupe Styled Coupe”).\n"
    "- Prefer real-world understanding of the model over any provided labels.\n"
    "- Output ONLY the final label in Title Case. No punctuation, no extra text.\n\n"
    "Examples (calibration only)\n"
    "- Audi A7 → Saloon Executive Sportback\n"
    "- BMW 5 Series GT → GT Large\n"
    "- BMW 4 Series Gran Coupe→ Gran Coupe Midsize\n"
    "- Porsche 911 → Sports Car Performance\n"
    "- Lexus LC 500 → Sports Car luxury\n"
    "- Ferrari 812 → Supercar\n"
    "- BMW X1 → SUV Compact\n"
    "- BMW X5 → SUV Large\n"
    "- VW Passat → Saloon Executive\n"
    "- Volvo V90 → Estate Large\n"
)


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


def call_gpt_label(years_text: str) -> Optional[str]:
    if not years_text or not str(years_text).strip():
        return ""
    try:
        from openai import OpenAI
    except Exception:
        return None

    _load_api_key_from_env_file()
    if not os.getenv("OPENAI_API_KEY"):
        return None

    client = OpenAI()
    # Use the provided prompt verbatim; provide only the series years as input context.
    user_content = f"{PROMPT_TEXT}\n\nInput (series production years only): {str(years_text).strip()}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
        )
        label = (resp.choices[0].message.content or "").strip()
        # Minimal post-clean: ensure no punctuation and Title Case already enforced by prompt
        return label
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Create 'Real Body Type' from series years using GPT-4o prompt")
    ap.add_argument("--input", default="Series + URL.xlsx")
    ap.add_argument("--output", default="Series + URL_with_real_body_type.xlsx")
    ap.add_argument("--years-col", default="Series (production years start-end)")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_excel(args.input, dtype=str)
    if args.years_col not in df.columns:
        raise SystemExit(f"Years column not found: {args.years_col}")

    if "Real Body Type" not in df.columns:
        df["Real Body Type"] = ""

    years_values: List[str] = df[args.years_col].astype(str).fillna("").tolist()

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {}
        for idx, yrs in enumerate(years_values):
            futures[ex.submit(call_gpt_label, yrs)] = idx
        done = 0
        total = len(futures)
        start = time.time()
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                val = fut.result()
            except Exception:
                val = None
            df.at[idx, "Real Body Type"] = val or ""
            done += 1
            if done == 1 or done % 25 == 0:
                rate = done / max(time.time() - start, 1e-6)
                eta = (total - done) / rate if rate > 0 else 0
                print(f"[gpt] {done}/{total} | {rate:.1f}/s | ETA ~{eta/60:.1f}m", flush=True)

    df.to_excel(args.output, index=False)
    print(f"Saved with 'Real Body Type': {args.output}")


if __name__ == "__main__":
    main()


