#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def find_column(df: pd.DataFrame, candidates: List[str], must_include: Optional[List[str]] = None) -> Optional[str]:
    must_include = [m.lower() for m in (must_include or [])]
    # 1) Exact match by candidates
    for name in candidates:
        if name in df.columns:
            return name
    # 2) Fuzzy by tokens
    def norm(s: str) -> List[str]:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip().split()
    for col in df.columns:
        tokens = set(norm(col))
        if all(m in tokens for m in must_include):
            return col
    return None


def extract_years(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    year_col = find_column(
        df,
        candidates=[
            "Series (production years start-end)",
            "Series_Production_Year",
            "Series prod year",
            "Series Production Year",
            "Series production year",
        ],
        must_include=["series", "production", "year"]
    )
    if not year_col:
        # Fallback: try any column that includes years-like patterns
        for col in df.columns:
            if re.search(r"\(.*year.*\)", col, flags=re.I):
                year_col = col
                break
    if not year_col:
        raise ValueError("Could not locate series years column")
    return year_col, df[year_col].astype(str).fillna("")


def pick_model_name(df: pd.DataFrame) -> str:
    candidates = [
        "Model", "Series", "Car", "Vehicle", "Model Name", "Series Name", "Make & Model",
        "Make and Model", "Name"
    ]
    col = find_column(df, candidates)
    if col:
        return col
    # Fallback to first textual column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return candidates[0]


def extract_bodytype(texts: List[str]) -> Optional[str]:
    body_keywords = [
        "hatchback", "saloon", "sedan", "estate", "wagon", "coupe", "gran coupe",
        "fastback", "suv", "crossover", "convertible", "roadster", "mpv"
    ]
    joined = " ".join([t.lower() for t in texts if isinstance(t, str)])
    for kw in body_keywords:
        if kw in joined:
            return kw
    return None


def find_text_col(df: pd.DataFrame, names: List[str], must_include_tokens: Optional[List[str]] = None) -> Optional[str]:
    col = find_column(df, candidates=names, must_include=must_include_tokens)
    if col:
        return col
    # Try fuzzy: match all tokens
    if must_include_tokens:
        for c in df.columns:
            tokens = re.sub(r"[^a-z0-9]+", " ", str(c).lower()).split()
            if all(t in tokens for t in must_include_tokens):
                return c
    return None


def find_numeric_col(df: pd.DataFrame, names: List[str], token_match: Optional[List[str]] = None) -> Optional[str]:
    col = find_text_col(df, names, token_match)
    if col:
        return col
    # Try fuzzy numeric columns with tokens
    if token_match:
        for c in df.columns:
            tokens = re.sub(r"[^a-z0-9]+", " ", str(c).lower()).split()
            if all(t in tokens for t in token_match):
                # Prefer numeric-like columns
                try:
                    pd.to_numeric(df[c].head(50), errors='coerce')
                    return c
                except Exception:
                    continue
    return None


def size_label_from_length(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
    length_col = find_column(df, candidates=["Length", "Length (mm)", "Overall length", "Length_mm"]) 
    if not length_col:
        return None, None
    try:
        series = pd.to_numeric(df[length_col], errors='coerce')
    except Exception:
        return None, None
    return length_col, series


def classify_size(bodytype: Optional[str], length_mm: Optional[float]) -> Optional[str]:
    if length_mm is not None:
        try:
            if length_mm >= 4900:
                return "full-size" if bodytype in ("suv", "crossover") else "executive"
            if length_mm >= 4800 and bodytype in ("suv", "crossover"):
                return "full-size"
            if length_mm >= 4600:
                return "large"
            if length_mm >= 4400:
                return "mid-size"
            return "compact"
        except Exception:
            pass
    # Fallback purely by bodytype
    if bodytype in ("suv", "crossover"):
        return "compact"
    if bodytype in ("saloon", "sedan", "estate", "wagon"):
        return "executive"
    return None


def extract_specs_from_text(text: str) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    if not isinstance(text, str):
        return specs
    t = text.lower()
    # Fuel type
    for fuel in ["diesel", "petrol", "hybrid", "plug-in hybrid", "phev", "electric"]:
        if fuel in t and "fuel" not in specs:
            specs["fuel"] = fuel
            break
    # Power
    m = re.search(r"(\d{2,4})\s*(?:bhp|hp)", t)
    if m:
        specs["power_hp"] = m.group(1)
    # 0-62 time
    m = re.search(r"0\s*[–-]\s*62\s*mph[^\d]*(\d{3}\.?\d?)s", t)
    if m:
        specs["zero_to_sixty_s"] = m.group(1)
    # Drivetrain
    for dt in ["rwd", "fwd", "awd", "4wd"]:
        if dt in t:
            specs["drivetrain"] = dt.upper()
            break
    # Transmission
    if "manual" in t:
        specs["trans"] = "manual"
    elif "automatic" in t or "auto" in t:
        specs["trans"] = "auto"
    # Economy
    m = re.search(r"(\d{2,3})\s*mpg", t)
    if m:
        specs["mpg"] = m.group(1)
    return specs


def extract_specs_from_table(row: pd.Series, df: pd.DataFrame) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    # engine/fuel
    engine_col = find_text_col(df, ["engine type", "engine", "Engine Type"], ["engine"]) or ""
    fuel_col = find_text_col(df, ["fuel type", "Fuel_Type", "Fuel"], ["fuel"]) or ""
    body_col = find_text_col(df, ["body type", "Body Type", "bodytype"], ["body"]) or ""

    engine = str(row.get(engine_col, "")).strip() if engine_col else ""
    fuel = str(row.get(fuel_col, "")).strip().lower() if fuel_col else ""
    body = str(row.get(body_col, "")).strip().lower() if body_col else ""

    if engine:
        specs["engine"] = engine
    if fuel:
        specs["fuel"] = fuel
    if body:
        specs["bodytype"] = body

    # power (hp)
    power_candidates = [
        find_numeric_col(df, ["Power (hp)", "Power", "bhp", "HP"], ["power"]),
        find_numeric_col(df, [], ["power", "hp"]),
    ]
    power_col = next((c for c in power_candidates if c), None)
    if power_col:
        try:
            power_val = pd.to_numeric(row.get(power_col, None), errors='coerce')
            if pd.notna(power_val):
                specs["power_hp"] = str(int(round(float(power_val))))
        except Exception:
            pass

    # 0-60 mph (use the fastest available if multiple exist)
    accel_cols = [c for c in df.columns if re.search(r"0\s*[-–]\s*60\s*mph", str(c), flags=re.I)]
    best_accel = None
    for c in accel_cols:
        try:
            val = pd.to_numeric(row.get(c, None), errors='coerce')
            if pd.notna(val):
                best_accel = val if best_accel is None else min(best_accel, val)
        except Exception:
            continue
    if best_accel is not None:
        specs["zero_to_sixty_s"] = f"{best_accel}"

    # seats
    seats_col = find_numeric_col(df, ["Seats"], ["seats"]) or ""
    if seats_col:
        try:
            seats_val = pd.to_numeric(row.get(seats_col, None), errors='coerce')
            if pd.notna(seats_val):
                specs["seats"] = str(int(seats_val))
        except Exception:
            pass

    # boot / luggage capacity (L)
    boot_col = None
    for c in df.columns:
        if re.search(r"(boot|luggage).*cap", str(c), flags=re.I):
            boot_col = c; break
    if boot_col:
        try:
            boot_val = pd.to_numeric(row.get(boot_col, None), errors='coerce')
            if pd.notna(boot_val):
                specs["boot_l"] = str(int(boot_val))
        except Exception:
            pass

    return specs


def build_specs_summary(model_name: str, years: str, bodytype: Optional[str], specs: Dict[str, str]) -> str:
    parts = []
    if specs.get("power_hp"):
        parts.append(f"{specs['power_hp']}hp")
    if specs.get("zero_to_sixty_s"):
        parts.append(f"0–62mph {specs['zero_to_sixty_s']}s")
    if specs.get("drivetrain"):
        parts.append(specs["drivetrain"])
    if specs.get("trans"):
        parts.append(specs["trans"])
    if specs.get("mpg"):
        parts.append(f"{specs['mpg']}mpg")
    if specs.get("boot_l"):
        parts.append(f"{specs['boot_l']}L boot")
    if specs.get("seats"):
        parts.append(f"{specs['seats']} seats")
    fuel = specs.get("fuel")
    body = bodytype if bodytype else ""
    core = ", ".join(parts) if parts else "key specs vary"
    out = f"{core}."
    if fuel:
        out = f"{fuel}, " + out
    if body:
        out = f"{body}, " + out
    return out.strip()


def parse_rivals(rivals_text: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if not isinstance(rivals_text, str) or not rivals_text.strip():
        return results
    text = rivals_text
    # Find name + rating first
    for m in re.finditer(r"([A-Za-z0-9][A-Za-z0-9\-&' ]+)\s*:\s*([\d.]+)/5", text):
        name = m.group(1).strip()
        rating = m.group(2).strip()
        # Try to capture a brief context after rating up to a separator
        tail = text[m.end(): m.end() + 120]
        ctx_match = re.search(r"\)?:\s*([^|]+?)(?=\||$)", tail)
        context = ctx_match.group(1).strip() if ctx_match else ""
        results.append({"name": name, "rating": f"{rating}/5", "context": context})
    # If nothing matched, try capturing URLs with names nearby
    if not results:
        for m in re.finditer(r"https?://\S+", text):
            url = m.group(0)
            # Look back 60 chars for a plausible name
            head = text[max(0, m.start()-60):m.start()]
            name_match = re.search(r"([A-Za-z][A-Za-z0-9\-&' ]{2,})$", head)
            name = name_match.group(1).strip() if name_match else ""
            results.append({"name": name, "rating": "", "context": "similar size and price"})
    return results[:5]


def essence_from_fields(model: str, years: str, bodytype: Optional[str], size_label: Optional[str], pros: str, cons: str,
                        make: str = "", engine: str = "", seats: str = "", boot_l: str = "", performance: str = "") -> str:
    # Keep within ~80–150 words by curating content length
    intro_bits = []
    if bodytype and size_label:
        intro_bits.append(f"a {size_label} {bodytype}")
    elif bodytype:
        intro_bits.append(f"a {bodytype}")
    name = f"{make} {model}".strip()
    intro = f"The {name} ({years}) is " + (" ".join(intro_bits) if intro_bits else "a capable option") + "."

    perf = "It offers a range of engines with balanced performance and refinement."
    practical = "Inside, it focuses on comfort and useful tech, with everyday practicality."

    if pros:
        perf = f"Strengths include {pros}."
    if cons:
        practical = f"Trade-offs: {cons}."

    verdict = "Best for buyers who want a sensible mix of usability and value."
    details = []
    if engine:
        details.append(engine)
    if performance:
        details.append(performance)
    if seats:
        details.append(f"{seats} seats")
    if boot_l:
        details.append(f"{boot_l}L boot")
    detail_sentence = (" It features " + ", ".join(details) + ".") if details else ""

    text = " ".join([intro, perf, practical + detail_sentence, verdict])
    # Trim to ~150 words
    words = text.split()
    if len(words) > 160:
        text = " ".join(words[:160])
    return text


def build_vectors(model: str, years: str, essence: str, rivals_json: List[Dict[str, str]], specs_summary: str, use_cases: str = "") -> Tuple[str, str, str]:
    # Main
    rival_contexts = ", ".join([r.get("context", "").strip() for r in rivals_json if r.get("context")])
    vt_main = f"{essence} Rivals: {rival_contexts}. Use cases: {use_cases}. Key facts: {specs_summary}.".strip()
    # Rivals
    def riv_str(r):
        name = r.get("name", "").strip()
        rating = r.get("rating", "").strip()
        ctx = r.get("context", "").strip()
        return f"{name} ({rating}): {ctx}".strip(": ")
    vt_rivals = f"{model} rivals: " + " | ".join([riv_str(r) for r in rivals_json if r.get("name")])
    # Specs
    vt_specs = f"{model} {years}. {specs_summary}"
    return vt_main, vt_rivals, vt_specs


def main():
    parser = argparse.ArgumentParser(description="Generate Essence paragraphs, structured Rivals, and vector text fields.")
    parser.add_argument("--input", default="Tabulated Pivots ONLY_enriched.xlsx", help="Input enriched Excel")
    parser.add_argument("--output", default=None, help="Output Excel path (default: <input>_with_essence.xlsx)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_with_essence.xlsx")

    df = pd.read_excel(in_path)

    years_col, years_series = extract_years(df)
    model_col = pick_model_name(df)

    pros_col = find_column(df, ["Pros"]) or "Pros"
    cons_col = find_column(df, ["Cons"]) or "Cons"
    article_col = find_column(df, ["Article_Text"]) or "Article_Text"
    rivals_col = find_column(df, ["Rivals"]) or "Rivals"
    make_col = find_text_col(df, ["Make"], ["make"]) or "Make"
    body_col = find_text_col(df, ["Body Type", "body type", "bodytype"], ["body"]) or ""
    engine_col = find_text_col(df, ["Engine Type", "engine type", "engine"], ["engine"]) or ""

    # Prepare helpers
    bodytype_guess = extract_bodytype([
        df.get(body_col, ""),
        df.get("Body", ""),
        df.get(article_col, "") if article_col in df else ""
    ])  # type: ignore
    length_col, length_series = size_label_from_length(df)

    essence_list = []
    rivals_json_list = []
    specs_summary_list = []
    vt_main_list = []
    vt_rivals_list = []
    vt_specs_list = []

    for i, row in df.iterrows():
        model = str(row.get(model_col, "")).strip()
        make = str(row.get(make_col, "")).strip()
        years = str(row.get(years_col, "")).strip()
        pros = str(row.get(pros_col, "")).strip()
        cons = str(row.get(cons_col, "")).strip()
        article = str(row.get(article_col, "")).strip()
        rivals_raw = row.get(rivals_col, "")
        engine = str(row.get(engine_col, "")).strip() if engine_col else ""

        # Size label per row
        bodytype = bodytype_guess or (str(row.get(body_col, "")).strip().lower() if body_col else None)
        length_val = None
        if length_series is not None:
            try:
                length_val = float(length_series.iloc[i]) if pd.notna(length_series.iloc[i]) else None
            except Exception:
                length_val = None
        size_label = classify_size(bodytype, length_val)

        # Specs summary using table data first, then text fallback
        specs = extract_specs_from_table(row, df)
        if not specs.get("fuel"):
            specs.update({k:v for k,v in extract_specs_from_text(article).items() if k not in specs})
        specs_summary = build_specs_summary(model, years, bodytype, specs)

        # Essence
        performance = ""
        if specs.get("power_hp") and specs.get("zero_to_sixty_s"):
            performance = f"{specs['power_hp']}hp, 0–62mph in {specs['zero_to_sixty_s']}s"
        seats = specs.get("seats", "")
        boot_l = specs.get("boot_l", "")
        essence = essence_from_fields(model, years, bodytype, size_label, pros, cons,
                                      make=make, engine=engine, seats=seats, boot_l=boot_l, performance=performance)

        # Rivals parsing
        rivals_json = parse_rivals(str(rivals_raw))

        # Vectors
        vt_main, vt_rivals, vt_specs = build_vectors(model, years, essence, rivals_json, specs_summary)

        essence_list.append(essence)
        rivals_json_list.append(rivals_json)
        specs_summary_list.append(specs_summary)
        vt_main_list.append(vt_main)
        vt_rivals_list.append(vt_rivals)
        vt_specs_list.append(vt_specs)

    # Apply to DataFrame
    df["Essence"] = essence_list
    df["Rivals_JSON"] = [str(r) for r in rivals_json_list]
    # Also create first three rival columns as flat columns (Name, Rating, Context)
    for idx in range(3):
        names, ratings, contexts = [], [], []
        for rivals in rivals_json_list:
            if idx < len(rivals):
                r = rivals[idx]
                names.append(r.get("name", ""))
                ratings.append(r.get("rating", ""))
                ctx = r.get("context", "") or "similar size and price"
                contexts.append(ctx)
            else:
                names.append("")
                ratings.append("")
                contexts.append("")
        df[f"Rival_{idx+1}_Name"] = names
        df[f"Rival_{idx+1}_Rating"] = ratings
        df[f"Rival_{idx+1}_Context"] = contexts

    df["Specs_Summary"] = specs_summary_list
    df["Vector_Text_Main"] = vt_main_list
    df["Vector_Text_Rivals"] = vt_rivals_list
    df["Vector_Text_Specs"] = vt_specs_list

    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


