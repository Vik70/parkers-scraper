#!/usr/bin/env python3
"""
Builds lifestyle-focused texts and IDs for the 13k pivot sheet:

- essence_family: 2–3 sentences at family (series-years) level
- essence_variant_delta: 1–2 sentences explaining engine/drivetrain feel
- rivals_json_family: list of {name, context} at family level
- family_id: normalized key (make_model_seriesYears)
- variant_id: lightweight per-row id (family_id + engine-ish)

Notes:
- No numbers in either text. Numbers can exist in sheet but are not inserted.
- Designed to run on the enriched sheet produced by merge_descriptions_to_pivots.py

Usage:
  python build_essence_and_variant_texts.py \
    --input "Tabulated Pivots ONLY_enriched_with_essence_v2.xlsx" \
    --output "Tabulated Pivots ONLY_enriched_with_essence_v2_texts.xlsx"
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ------------------------------
# Column helpers
# ------------------------------

def _norm_tokens(s: str) -> List[str]:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip().split()


def find_column(df: pd.DataFrame, candidates: Iterable[str], must_include: Optional[Iterable[str]] = None) -> Optional[str]:
    """Find a column by exact name or by token inclusion."""
    for name in candidates:
        if name in df.columns:
            return name
    if must_include:
        must = set(t.lower() for t in must_include)
        for col in df.columns:
            toks = set(_norm_tokens(str(col)))
            if must.issubset(toks):
                return col
    return None


def find_text_col(df: pd.DataFrame, names: List[str], must_tokens: Optional[List[str]] = None) -> Optional[str]:
    col = find_column(df, names, must_tokens)
    if col:
        return col
    if must_tokens:
        needed = set(must_tokens)
        for c in df.columns:
            toks = set(_norm_tokens(str(c)))
            if needed.issubset(toks):
                return c
    return None


def find_years_col(df: pd.DataFrame) -> Optional[str]:
    return find_column(
        df,
        [
            "Series (production years start-end)",
            "Series_Production_Year",
            "Series prod year",
            "Series production year",
            "Series Production Year",
        ],
        ["series", "production", "year"],
    )


def normalize_key(value) -> str:
    s = "" if value is None else str(value)
    s = re.sub(r"\s+", " ", s.strip()).lower()
    return s


# ------------------------------
# Family + size classification
# ------------------------------

def classify_size_bucket(body_type: Optional[str], length_mm: Optional[float]) -> Optional[str]:
    """Map to a coarse size bucket. Numbers guide the mapping; we do not insert them into text."""
    if body_type:
        bt = body_type.lower()
    else:
        bt = ""

    if length_mm is not None:
        try:
            l = float(length_mm)
        except Exception:
            l = None
    else:
        l = None

    if l is not None:
        # Simple, robust thresholds; not printed anywhere
        if l >= 4900:
            return "large SUV" if "suv" in bt or "crossover" in bt else "exec saloon"
        if l >= 4600:
            return "large SUV" if "suv" in bt or "crossover" in bt else "large"
        if l >= 4400:
            return "crossover SUV" if "suv" in bt or "crossover" in bt else "medium"
        # below ~4.4m → small/compact
        return "small hatchback" if "hatch" in bt else ("crossover SUV" if "suv" in bt else "compact")

    # Fallback purely by body type
    if any(k in bt for k in ["suv", "crossover"]):
        return "crossover SUV"
    if any(k in bt for k in ["saloon", "sedan"]):
        return "small saloon"
    if "estate" in bt or "wagon" in bt:
        return "estate"
    if "hatch" in bt:
        return "small hatchback"
    if "mpv" in bt:
        return "mpv"
    if "coupe" in bt:
        return "coupe"
    return None


def pick_use_case(body_type: Optional[str]) -> str:
    bt = (body_type or "").lower()
    if any(k in bt for k in ["suv", "crossover", "estate", "wagon"]):
        return "family trips and everyday versatility"
    if any(k in bt for k in ["saloon", "sedan"]):
        return "comfortable commuting and long-distance cruising"
    if "mpv" in bt:
        return "family hauling and practicality"
    if "coupe" in bt:
        return "weekend drives and style-focused buyers"
    if "hatch" in bt:
        return "city duties and easy daily use"
    return "general everyday driving"


def pick_highlights(body_type: Optional[str]) -> List[str]:
    bt = (body_type or "").lower()
    if any(k in bt for k in ["suv", "crossover"]):
        return ["space and visibility", "refined ride"]
    if any(k in bt for k in ["saloon", "sedan"]):
        return ["comfort and refinement", "premium feel"]
    if any(k in bt for k in ["estate", "wagon"]):
        return ["big boot and practicality", "long-trip comfort"]
    if "mpv" in bt:
        return ["clever interior", "family-friendly usability"]
    if "coupe" in bt:
        return ["style and engagement"]
    if "hatch" in bt:
        return ["easy to park", "efficient running"]
    return ["easy to live with"]


# ------------------------------
# Variant delta rules
# ------------------------------

@dataclass
class VariantSignals:
    fuel_type: str
    engine_text: str
    transmission: str
    drivetrain: str


def extract_variant_signals(engine: str, fuel: str, transmission: str, drivetrain: str) -> VariantSignals:
    return VariantSignals(
        fuel_type=(fuel or "").lower(),
        engine_text=(engine or "").lower(),
        transmission=(transmission or "").lower(),
        drivetrain=(drivetrain or "").lower(),
    )


def describe_variant_delta(sig: VariantSignals) -> str:
    e = sig.engine_text
    f = sig.fuel_type
    t = sig.transmission
    d = sig.drivetrain

    lines: List[str] = []

    # Fuel/engine family
    if any(k in e for k in ["tdi", "dci", "cdi"]) or f == "diesel":
        lines.append("suited to long-distance efficiency and relaxed motorway cruising")
    elif any(k in e for k in ["phev", "plug-in"]) or f in ("phev", "plug-in hybrid"):
        lines.append("great for low running costs and tax, especially around town")
    elif "hybrid" in e or f == "hybrid":
        lines.append("quiet and frugal in stop-start driving")
    elif any(k in e for k in ["ev", "electric"]) or f in ("ev", "electric"):
        lines.append("very quiet with instant response; best if you can charge at home")
    elif re.search(r"\b(?:m|amg|rs|vrs|type\s*r|sti|gti|p\s*tuned)\b", e):
        lines.append("genuinely quick and more enthusiast-leaning")
    elif re.search(r"\b(1\.0|1\.2|1\.3|1\.4)\b|\b(110|120|125)\b", e):
        lines.append("nippy and economical for city-friendly use")
    elif re.search(r"\b(1\.5|1\.6|1\.8|2\.0)\b", e):
        lines.append("balanced for daily driving, mixing pace and efficiency")
    else:
        if f == "petrol":
            lines.append("responsive and easy-going for everyday use")

    # Transmission nuance
    if any(k in t for k in ["dct", "dsg", "auto", "tronic"]):
        lines.append("smoother shifting for hassle-free driving")
    elif "manual" in t:
        lines.append("more engaging feel if you like to be involved")

    # Drivetrain
    if any(k in d for k in ["quattro", "xdrive", "4matic", "awd", "4wd"]):
        lines.append("added all-weather confidence")

    # Compose 1–2 concise sentences
    if not lines:
        return "A sensible choice for typical daily driving."
    first = lines[0]
    rest = ", ".join(lines[1:])
    if rest:
        return f"This variant is {first}, with {rest}."
    return f"This variant is {first}."


# ------------------------------
# Rivals parsing and aggregation
# ------------------------------

def parse_rivals(rivals_text: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if not isinstance(rivals_text, str) or not rivals_text.strip():
        return results
    text = rivals_text
    for m in re.finditer(r"([A-Za-z0-9][A-Za-z0-9\-&' ]+)\s*:\s*([\d.]+)/5", text):
        name = m.group(1).strip()
        # take small context tail after rating
        tail = text[m.end() : m.end() + 120]
        ctx_match = re.search(r"\)?:\s*([^|]+?)(?=\||$)", tail)
        context = ctx_match.group(1).strip() if ctx_match else ""
        results.append({"name": name, "context": context})
    if not results:
        # Try names near URLs
        for m in re.finditer(r"https?://\S+", text):
            head = text[max(0, m.start() - 60) : m.start()]
            name_match = re.search(r"([A-Za-z][A-Za-z0-9\-&' ]{2,})$", head)
            name = name_match.group(1).strip() if name_match else ""
            if name:
                results.append({"name": name, "context": "similar size and price"})
    # Deduplicate by name, preserve order
    seen = set()
    deduped: List[Dict[str, str]] = []
    for r in results:
        nm = r.get("name", "").strip().lower()
        if nm and nm not in seen:
            deduped.append({"name": r.get("name", "").strip(), "context": r.get("context", "").strip()})
            seen.add(nm)
    return deduped[:5]


def aggregate_family_rivals(df: pd.DataFrame, family_id_col: str, rivals_col: Optional[str]) -> Dict[str, List[Dict[str, str]]]:
    fam_to_rivals: Dict[str, List[Dict[str, str]]] = {}
    if not rivals_col or rivals_col not in df.columns:
        return fam_to_rivals
    for _, row in df.iterrows():
        fam = str(row[family_id_col])
        parsed = parse_rivals(str(row.get(rivals_col, "")))
        if not parsed:
            continue
        acc = fam_to_rivals.setdefault(fam, [])
        for r in parsed:
            name_key = r.get("name", "").strip().lower()
            if not name_key:
                continue
            if all(name_key != x.get("name", "").strip().lower() for x in acc):
                acc.append(r)
        fam_to_rivals[fam] = acc[:5]
    return fam_to_rivals


# ------------------------------
# Essence builders
# ------------------------------

def build_essence_family(make: str, model: str, years: str, body_type: Optional[str], size_bucket: Optional[str]) -> str:
    name = f"{make} {model}".strip()
    intro_shape = f"a {size_bucket} {body_type}" if (size_bucket and body_type) else (f"a {body_type}" if body_type else "a versatile option")
    intro = f"The {name} ({years}) is {intro_shape}."

    use_case = pick_use_case(body_type)
    highlights = pick_highlights(body_type)
    good_bits = ", ".join(highlights[:2])
    mid = f"It suits {use_case}, with emphasis on {good_bits}."

    outro = "Stands out as easy to live with and broadly appealing."

    return " ".join([intro, mid, outro]).strip()


# ------------------------------
# Main CLI
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate essence_family, essence_variant_delta, rivals_json_family, family_id, variant_id")
    parser.add_argument("--input", default="Tabulated Pivots ONLY_enriched_with_essence_v2.xlsx", help="Input Excel path")
    parser.add_argument("--output", default=None, help="Output Excel path (default: <input>_texts.xlsx)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_texts.xlsx")

    df = pd.read_excel(in_path)

    years_col = find_years_col(df)
    if not years_col:
        raise SystemExit("Could not locate 'series production year' column")

    make_col = find_text_col(df, ["Make"], ["make"]) or "Make"
    model_col = find_text_col(df, ["Model", "Series", "Model Name", "Make & Model", "Name"], None) or "Model"
    body_col = find_text_col(df, ["Body Type", "body type", "Body", "bodytype"], ["body"]) or ""
    length_col = find_text_col(df, ["Length", "Length (mm)", "Overall length", "Length_mm"], None) or ""
    engine_col = find_text_col(df, ["Engine", "Engine Type", "engine type"], ["engine"]) or ""
    fuel_col = find_text_col(df, ["Fuel", "Fuel Type", "Fuel_Type"], ["fuel"]) or ""
    trans_col = find_text_col(df, ["Transmission", "Gearbox"], None) or ""
    drive_col = find_text_col(df, ["Drivetrain", "Drive", "Drive Type"], None) or ""
    rivals_col = find_text_col(df, ["Rivals"], None)

    # Build IDs
    make_vals = df.get(make_col, "").astype(str)
    model_vals = df.get(model_col, "").astype(str)
    years_vals = df.get(years_col, "").astype(str)

    family_id: List[str] = []
    for mk, md, yr in zip(make_vals, model_vals, years_vals):
        fid = normalize_key(f"{mk} {md} {yr}")
        family_id.append(fid)
    df["family_id"] = family_id

    # Variant id: family + terse engine token
    engines = df.get(engine_col, "").astype(str) if engine_col else pd.Series([""] * len(df))
    engine_token = engines.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip().str.replace(r"\s+", "-", regex=True)
    df["variant_id"] = [f"{fid}__{tok}" if tok else f"{fid}__row{i}" for i, (fid, tok) in enumerate(zip(df["family_id"], engine_token))]

    # Pre-compute per-row body/length for family summaries
    body_vals = df.get(body_col, "").astype(str) if body_col else pd.Series([""] * len(df))
    length_vals = None
    if length_col and length_col in df.columns:
        try:
            length_vals = pd.to_numeric(df[length_col], errors="coerce")
        except Exception:
            length_vals = None

    # Build essence_family per family_id using the first representative row
    fam_first_index: Dict[str, int] = {}
    for idx, fid in enumerate(df["family_id" ]):
        if fid not in fam_first_index:
            fam_first_index[fid] = idx

    essence_family_map: Dict[str, str] = {}
    for fid, idx in fam_first_index.items():
        mk = str(make_vals.iloc[idx]).strip()
        md = str(model_vals.iloc[idx]).strip()
        yr = str(years_vals.iloc[idx]).strip()
        bt = str(body_vals.iloc[idx]).strip().lower() if body_col else ""
        ln = float(length_vals.iloc[idx]) if (length_vals is not None and pd.notna(length_vals.iloc[idx])) else None
        size_bucket = classify_size_bucket(bt, ln)
        essence_family_map[fid] = build_essence_family(mk, md, yr, bt or None, size_bucket)

    df["essence_family"] = [essence_family_map[fid] for fid in df["family_id"]]

    # Aggregate rivals per family
    fam_to_rivals = aggregate_family_rivals(df, "family_id", rivals_col)
    df["rivals_json_family"] = [json.dumps(fam_to_rivals.get(fid, []), ensure_ascii=False) for fid in df["family_id"]]

    # Variant delta per row
    fuel_vals = df.get(fuel_col, "").astype(str) if fuel_col else pd.Series([""] * len(df))
    trans_vals = df.get(trans_col, "").astype(str) if trans_col else pd.Series([""] * len(df))
    drive_vals = df.get(drive_col, "").astype(str) if drive_col else pd.Series([""] * len(df))

    deltas: List[str] = []
    for eng, fuel, tr, drv in zip(engines, fuel_vals, trans_vals, drive_vals):
        sig = extract_variant_signals(eng, fuel, tr, drv)
        deltas.append(describe_variant_delta(sig))
    df["essence_variant_delta"] = deltas

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


