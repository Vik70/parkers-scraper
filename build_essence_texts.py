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


def choose_size_descriptor(body_type: str, length_mm: float | None) -> str:
    bt = norm(body_type)
    L = None
    try:
        L = float(length_mm) if length_mm is not None else None
    except Exception:
        L = None
    # Sportback / fastback / gran coupe → executive hatchback; size by length
    if any(k in bt for k in ("sportback", "fastback", "gran coupe", "gran-coupe", "grancoupe")):
        if L is not None and L >= 4850:
            return "large executive hatchback (gran coupe style)"
        return "executive hatchback (gran coupe style)"
    if "estate" in bt or "touring" in bt:
        return "premium estate" if any(k in bt for k in ("premium",)) else "estate"
    if "suv" in bt or "crossover" in bt:
        return "large SUV" if (L is not None and L >= 4700) else "crossover SUV"
    if "saloon" in bt or "sedan" in bt:
        return "executive saloon" if (L is None or L >= 4700) else "small saloon"
    if "hatch" in bt:
        if L is not None:
            return "medium hatchback" if L >= 4300 else "small hatchback"
        return "medium hatchback"
    if "mpv" in bt:
        return "MPV"
    if "coupe" in bt:
        return "coupe"
    # Fallback purely by length
    if L is not None and L >= 4850:
        return "large executive car"
    return body_type or "car"


def performance_bucket(zero_to_62_s: float | None) -> str:
    if zero_to_62_s is None:
        return "unknown"
    t = float(zero_to_62_s)
    if t < 3.6:
        return "hyper-performance"
    if t < 5.0:
        return "serious performance"
    if t < 7.0:
        return "brisk"
    if t < 10.0:
        return "relaxed"
    return "economical"


def mpg_bucket(mpg: float | None) -> str:
    if mpg is None:
        return ""
    v = float(mpg)
    if v >= 55:
        return "strong economy"
    if v >= 45:
        return "good economy"
    if v >= 35:
        return "average economy"
    return "lower economy"


def boot_bucket(boot_l: float | None) -> str:
    if boot_l is None:
        return ""
    b = float(boot_l)
    if b < 300:
        return "pack light"
    if b < 450:
        return "usable hatch practicality"
    if b < 600:
        return "generous boot capacity"
    return "extra-large cargo space"


def parse_first_numeric(df: pd.DataFrame, row: pd.Series, patterns: tuple[str, ...]) -> float | None:
    for col in df.columns:
        name = str(col)
        if any(re.search(p, name, flags=re.I) for p in patterns):
            try:
                val = pd.to_numeric(row.get(col, None), errors="coerce")
                if pd.notna(val):
                    return float(val)
            except Exception:
                continue
    return None


def buyer_fit_from_signals(fuel: str, perf_bucket: str, body_type: str, mpg_b: str) -> str:
    f = norm(fuel)
    bt = norm(body_type)
    fits: list[str] = []
    if any(k in f for k in ("ev", "electric", "phev", "plug")):
        fits.append("eco-conscious commuters")
    if any(k in f for k in ("diesel", "tdi", "dci", "cdi")) or mpg_b in ("strong economy", "good economy"):
        fits.append("high-mileage professionals")
    if perf_bucket not in ("hyper-performance", "serious performance") and any(k in bt for k in ("estate", "touring", "suv", "crossover")):
        fits.append("practical families")
    if perf_bucket in ("hyper-performance", "serious performance"):
        fits.append("performance enthusiasts")
    if "hatch" in bt and perf_bucket in ("relaxed", "economical"):
        fits.append("first-time buyers")
    if not fits:
        fits.append("city and suburban commuters")
    # De-duplicate while preserving order
    seen = set()
    ordered = [x for x in fits if not (x in seen or seen.add(x))]
    return ", ".join(ordered[:2])


def _clean_years(model: str, years: str) -> str:
    s = str(years)
    # Prefer extracting year range or start year + onwards
    m = re.findall(r"(19\d{2}|20\d{2})", s)
    if len(m) >= 2:
        return f"{m[0]}–{m[1]}"
    if len(m) == 1:
        return f"{m[0]}–present"
    # Fallback: strip duplicated model prefix if present
    s2 = s.replace(str(model), "").strip()
    return s2 or s


def compose_essence_main(make: str, model: str, years: str, body_type: str, engine: str, trans: str,
                         zero_to_62_s: float | None, mpg: float | None, boot_l: float | None, seats: float | None,
                         length_mm: float | None) -> str:
    years_clean = _clean_years(model, years)
    size_desc = choose_size_descriptor(body_type, length_mm)
    intro = f"The {make} {model} ({years_clean}) is a {size_desc}."

    # Driving character from engine/trans + performance
    e = norm(engine)
    t = norm(trans)
    perf_b = performance_bucket(zero_to_62_s)
    phrases: list[str] = []
    if any(k in e for k in ("phev", "plug-in")):
        phrases.append("quiet, tax-friendly running in town on electric power")
    elif "hybrid" in e:
        phrases.append("quiet, efficient progress in stop-start driving")
    elif any(k in e for k in ("ev", "electric")):
        phrases.append("silent, instant response, best with home charging")
    elif any(k in e for k in ("diesel", "tdi", "dci", "cdi")):
        phrases.append("relaxed long‑distance cruising with strong economy")
    elif re.search(r"\b(1\.0|1\.2|1\.3|1\.4)\b", e):
        phrases.append("nippy, efficient feel that suits urban and suburban use")
    elif re.search(r"\b(1\.5|1\.6|1\.8|2\.0)\b", e):
        phrases.append("balanced everyday pace with sensible running costs")

    if any(k in t for k in ("auto", "dsg", "dct", "tronic")):
        phrases.append("an effortless automatic to keep things smooth")
    elif "manual" in t:
        phrases.append("a more engaging manual shift")

    # Performance bucket tone
    perf_map = {
        "hyper-performance": "ferociously quick with explosive acceleration",
        "serious performance": "genuinely brisk when you want to push on",
        "brisk": "brisk everyday pace",
        "relaxed": "relaxed but capable",
        "economical": "focused on economy over outright speed",
    }
    if perf_b != "unknown":
        phrases.append(perf_map[perf_b])

    driving = " It delivers " + ", ".join(phrases) + "." if phrases else ""

    # Practicality & interior
    bb = boot_bucket(boot_l)
    seats_txt = "" if seats is None else f"{int(seats)} seats"
    practicality_bits = []
    if bb:
        practicality_bits.append(bb)
    if seats_txt:
        practicality_bits.append(seats_txt)
    practicality = " The boot and seating offer " + (", ".join(practicality_bits)) + "." if practicality_bits else ""

    # Highlights
    mb = mpg_bucket(mpg)
    highlights = []
    if mb:
        highlights.append(mb)
    highlights_txt = " Key highlights: " + ", ".join(highlights) + "." if highlights else ""

    # Buyer fit
    buyer_fit = buyer_fit_from_signals(engine, perf_b, body_type, mb)
    buyer = f" Best suited for: {buyer_fit}."

    text = (intro + driving + practicality + highlights_txt + buyer).strip()
    # Trim to ~150 words
    words = text.split()
    if len(words) > 150:
        text = " ".join(words[:150])
    return text


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

    # Build per-variant essence_main with engine/trans baked in and buckets
    essence_main_list: list[str] = []
    delta_list: list[str] = []
    specs_compact_list: list[str] = []
    for _, row in df.iterrows():
        make = row.get(args.make_col, "")
        model = row.get(args.model_col, "")
        yrs = row.get(args.years_col, "")
        body = row.get(args.body_col, "")
        engine = row.get(args.engine_col, "")
        trans = row.get(args.trans_col, "")

        zero_to_62 = parse_first_numeric(df, row, (r"0\s*[-–]?\s*60\s*mph", r"0\s*[-–]?\s*62"))
        mpg_val = parse_first_numeric(df, row, (r"mpg", r"wltp", r"economy"))
        boot_val = parse_first_numeric(df, row, (r"luggage", r"boot", r"capacity.*litre"))
        seats_val = parse_first_numeric(df, row, (r"seats",))
        length_val = parse_first_numeric(df, row, (r"length\s*\(mm\)", r"length_mm", r"average of length"))

        essence_main_list.append(
            compose_essence_main(make, model, yrs, body, engine, trans, zero_to_62, mpg_val, boot_val, seats_val, length_val)
        )

        # Keep delta as ancillary (optional)
        delta_list.append(variant_delta_from_specs(engine, row.get(args.fuel_col, ""), trans, row.get(args.drive_col, "")))

        # Specs summary compact (numbers OK)
        parts = []
        if zero_to_62 is not None:
            parts.append(f"0–62mph {zero_to_62:.1f}s")
        if isinstance(trans, str) and trans:
            parts.append(trans)
        fuel = row.get(args.fuel_col, "")
        if isinstance(fuel, str) and fuel:
            parts.append(str(fuel))
        if seats_val is not None:
            parts.append(f"{int(seats_val)} seats")
        if boot_val is not None:
            parts.append(f"{int(round(boot_val))}L boot")
        if mpg_val is not None:
            parts.append(f"{int(round(mpg_val))}mpg")
        specs_compact_list.append("; ".join(parts))

    df["essence_main"] = essence_main_list
    df["essence_variant_delta"] = delta_list
    df["specs_summary_compact"] = specs_compact_list

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


