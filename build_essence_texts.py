#!/usr/bin/env python3
"""
build_essence_texts.py

Generate user-focused descriptions for the 13k sheet:
- essence_family (2–3 sentences)  ← once per (make, model, series_years)
- essence_variant_delta (1–2 sentences) ← per engine variant

Modes:
  --mode rules        (default; deterministic, no API cost)
  --mode llm          (OpenAI-assisted family copy; cached per family; rules fallback)

Usage examples:
  python build_essence_texts.py --input 13k.xlsx --output 13k_with_essence.xlsx --mode rules \
    --make-col Make --model-col Model --years-col "Series (production years start-end)" \
    --body-col "Body Type" --engine-col Engine --fuel-col Fuel_Type --trans-col Transmission --drive-col Drivetrain

  # LLM-assisted
  OPENAI_API_KEY=... python build_essence_texts.py --input 13k.xlsx --output 13k_with_essence.xlsx --mode llm [...cols]
"""

import argparse
import json
import os
import re
from typing import Dict

import pandas as pd


# ---------------- helpers ----------------

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", ("" if s is None else str(s)).strip().lower())


def build_family_id(make: str, model: str, years: str) -> str:
    return f"{norm(make)}|{norm(model)}|{norm(years)}"


def size_bucket(body_type: str, length_mm=None) -> str:
    bt = norm(body_type)
    # quick rules; can be extended with thresholds by length_mm
    try:
        L = float(length_mm) if length_mm is not None and str(length_mm).strip() else None
    except Exception:
        L = None
    if "hatch" in bt:
        return "small hatchback" if (L is None or L < 4300) else "medium hatchback"
    if "estate" in bt or "touring" in bt:
        return "estate"
    if "suv" in bt or "crossover" in bt:
        return "crossover SUV" if (L is None or L < 4600) else "large SUV"
    if "gran coupe" in bt or ("coupe" in bt and "gran" in bt):
        return "gran coupe"
    if "coupe" in bt:
        return "coupe"
    if "saloon" in bt or "sedan" in bt:
        return "small saloon" if (L is None or L < 4700) else "executive saloon"
    if "mpv" in bt:
        return "MPV"
    return body_type or "car"


ENGINE_PERF_TOKENS = re.compile(r"\b(m|amg|rs|vrs|sti|type[\s-]?r|gti|cupra|quadrifoglio|s\/?|\br\/?)(?:\b|[^a-z])", re.I)


def variant_delta_from_specs(engine: str, fuel: str, trans: str, drive: str) -> str:
    e = norm(engine)
    f = norm(fuel)
    t = norm(trans)
    d = norm(drive)

    notes = []

    perf = bool(ENGINE_PERF_TOKENS.search(engine or ""))
    small_turbo = bool(re.search(r"\b(0\.9|1\.0|1\.2|1\.3|1\.4)\b|(\b1[0-4]\d?\s?t)|\b(110|120)\s?ps\b", e))
    mid_petrol = ("1.5" in e or "1.6" in e or "1.8" in e or "2.0" in e) and (
        "t" in e or "tsi" in e or "tfs" in e or "turbo" in e or "na" in e
    )
    big_engine = bool(re.search(r"\b(3\.\d|4\.\d|V6|V8|V12)\b", engine or "", re.I))

    if "diesel" in f or re.search(r"\btdi|dci|cdi|hdi|cdti\b", e):
        notes.append("great for long motorway journeys with relaxed cruising and strong economy")
    if "hybrid" in f and "plug" not in f:
        notes.append("suited to mixed driving with lower running costs in town")
    if "plug" in f or "phev" in f:
        notes.append("ideal for short commutes with low tax and the ability to drive on electric power in town")
    if "ev" in f or "electric" in f:
        notes.append("very quiet and smooth with instant response; best if you can home-charge")
    if perf or big_engine:
        notes.append("genuinely quick and aimed at enthusiasts")
    if small_turbo and not perf:
        notes.append("nippy and efficient, easy to live with in the city")
    if mid_petrol and not perf:
        notes.append("a balanced choice mixing pace and efficiency for daily use")

    if any(k in t for k in ("auto", "dsg", "dct", "tronic")):
        notes.append("smoother automatic gearbox for effortless driving")
    if "manual" in t:
        notes.append("manual gearbox for a more engaging feel")
    if any(k in d for k in ("awd", "4matic", "xdrive", "quattro")):
        notes.append("added confidence from all-wheel drive in poor weather")

    if not notes:
        notes.append("a straightforward, easygoing choice for everyday driving")

    return " ".join([notes[0]] + ([notes[1]] if len(notes) > 1 else []))


def family_essence_rules(make: str, model: str, series_years: str, body_type: str, length_mm=None) -> str:
    size = size_bucket(body_type, length_mm)
    brand_hint = {
        "bmw": "driver-focused feel with premium refinement",
        "audi": "tech-led interior and solid refinement",
        "mercedes-benz": "comfort and luxury first",
        "mercedes": "comfort and luxury first",
        "vw": "all-round capability and quality",
        "volkswagen": "all-round capability and quality",
        "skoda": "space and value",
        "seat": "youthful, sporty vibe",
        "kia": "strong value and long warranties",
        "hyundai": "value, warranty and ease of use",
        "toyota": "reliability and efficiency",
        "honda": "usability and longevity",
        "mazda": "driver involvement with efficiency",
    }.get(norm(make), "a well-rounded package")

    audience = (
        "ideal for city use and new drivers" if "small hatchback" in size else
        "great for families and daily commuting" if ("medium hatchback" in size or "estate" in size or "crossover" in size) else
        "well-suited to professionals and long-distance commuters" if ("executive" in size or "saloon" in size or "gran coupe" in size) else
        "best for those who value style and weekend enjoyment" if "coupe" in size else
        "for those needing maximum space and versatility" if "mpv" in size else
        "a versatile choice for a wide range of drivers"
    )

    good_bits = {
        "small hatchback": "easy to park, low running costs",
        "medium hatchback": "good balance of space and efficiency",
        "estate": "huge boot and family practicality",
        "crossover SUV": "higher driving position and family-friendly usability",
        "large SUV": "spacious interior and long-distance comfort",
        "executive saloon": "refinement, comfort and a premium cabin",
        "small saloon": "comfort and a more grown-up feel than a hatchback",
        "gran coupe": "sleek looks with more practicality than a traditional coupe",
        "coupe": "standout styling and an engaging drive",
        "MPV": "ultimate people-moving practicality",
    }.get(size, "a sensible mix of comfort, usability and value")

    return (
        f"The {make} {model} ({series_years}) is a {size}. "
        f"It’s {audience}. "
        f"Strengths include {good_bits} and {brand_hint}."
    )


def llm_family_essence(make: str, model: str, series_years: str, body_type: str, length_mm=None) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI()
    prompt = f"""
Write 2–3 sentences describing the {make} {model} ({series_years}) as a product someone would choose.
Rules:
- Start with its size/style (e.g., "small hatchback", "executive saloon", "crossover SUV").
- Say the primary use case, who would drive it, and what’s good about it.
- Optional notable comment (image/reputation).
- No specs or numbers. Plain, friendly tone.
Body type: {body_type}; Length(mm): {length_mm or 'n/a'}.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# ---------------- main ----------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Build essence_family and essence_variant_delta texts for 13k sheet")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["rules", "llm"], default="rules")
    ap.add_argument("--make-col", default="Make")
    ap.add_argument("--model-col", default="Model")
    ap.add_argument("--years-col", default="Series (production years start-end)")
    ap.add_argument("--body-col", default="Body Type")
    ap.add_argument("--len-col", default="Length_mm")
    ap.add_argument("--engine-col", default="Engine")
    ap.add_argument("--fuel-col", default="Fuel_Type")
    ap.add_argument("--trans-col", default="Transmission")
    ap.add_argument("--drive-col", default="Drivetrain")
    args = ap.parse_args()

    df = pd.read_excel(args.input)
    # Map existing rivals JSON if present
    if "rivals_json_family" not in df.columns and "Rivals_JSON" in df.columns:
        df["rivals_json_family"] = df["Rivals_JSON"].astype(str)

    # Build family_id
    df["family_id"] = [
        build_family_id(row.get(args.make_col), row.get(args.model_col), row.get(args.years_col))
        for _, row in df.iterrows()
    ]

    # One essence per family
    fam_text_map: Dict[str, str] = {}
    rep_rows = df.groupby("family_id", dropna=False).first().reset_index()
    for _, r in rep_rows.iterrows():
        make = r.get(args.make_col)
        model = r.get(args.model_col)
        yrs = r.get(args.years_col)
        body = r.get(args.body_col)
        length = r.get(args.len_col)
        if args.mode == "llm":
            try:
                text = llm_family_essence(make, model, yrs, body, length)
            except Exception:
                text = family_essence_rules(make, model, yrs, body, length)
        else:
            text = family_essence_rules(make, model, yrs, body, length)
        fam_text_map[r["family_id"]] = text

    df["essence_family"] = df["family_id"].map(fam_text_map)

    # Variant deltas per row
    def build_delta(row):
        return variant_delta_from_specs(
            row.get(args.engine_col),
            row.get(args.fuel_col),
            row.get(args.trans_col),
            row.get(args.drive_col),
        )

    df["essence_variant_delta"] = df.apply(build_delta, axis=1)

    # Ensure rivals_json_family column exists
    if "rivals_json_family" not in df.columns:
        df["rivals_json_family"] = "[]"

    df.to_excel(args.output, index=False)
    print(f"✅ Wrote: {args.output}")
    print(
        "Coverage:",
        f"essence_family nulls={df['essence_family'].isna().sum()}",
        f"essence_variant_delta nulls={df['essence_variant_delta'].isna().sum()}",
    )


if __name__ == "__main__":
    main()


