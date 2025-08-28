import argparse
import os
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


CANONICAL_LABELS = [
    "small hatchback",
    "medium hatchback",
    "estate",
    "crossover SUV",
    "large SUV",
    "small saloon",
    "executive saloon",
    "coupe",
    "gran coupe",
    "executive hatchback (gran coupe style)",
    "convertible",
    "MPV",
    "pickup",
]


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


def call_llm(make: str, model: str, years: str, raw_body: str, dims: str) -> Optional[str]:
    try:
        from openai import OpenAI
    except Exception:
        return None

    _load_api_key_from_env_file()
    if not os.getenv("OPENAI_API_KEY"):
        return None

    client = OpenAI()
    system = (
        "You are an automotive body style classifier. Return ONE label from the allowed set only. "
        "Be strict and concise. If the source label is vague or wrong, use common sense by model name "
        "and dimensions to choose the best fit. Allowed labels: " + ", ".join(CANONICAL_LABELS) + "."
    )
    user = (
        f"Make/Model: {make} {model} ({years}).\n"
        f"Raw body type: {raw_body or 'n/a'}.\n"
        f"Clues (dimensions/spec hints may be sparse): {dims or 'n/a'}.\n"
        "Respond with only one of the allowed labels, no punctuation."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        text = text.replace("hatch back", "hatchback")
        if text in CANONICAL_LABELS:
            return text
        if "gran" in text and "coupe" in text:
            return "gran coupe"
        if "executive" in text and ("hatch" in text or "gran" in text):
            return "executive hatchback (gran coupe style)"
        if "saloon" in text or "sedan" in text:
            return "executive saloon" if "executive" in text else "small saloon"
        if "estate" in text or "touring" in text:
            return "estate"
        if "mpv" in text:
            return "MPV"
        if "convertible" in text or "cabrio" in text or "roadster" in text:
            return "convertible"
        if "coupe" in text:
            return "coupe"
        if "suv" in text or "crossover" in text:
            return "large SUV" if "large" in text else "crossover SUV"
        if "hatch" in text:
            return "medium hatchback"
        return None
    except Exception:
        return None


def dims_context_from_row(row: pd.Series) -> str:
    fields = [
        ("length_mm", ("length", "length_mm", "Average of Length (mm)", "Length_mm")),
        ("width_mm", ("width", "width_mm", "Average of Width (mm)", "Width_mm")),
        ("height_mm", ("height", "height_mm", "Average of Height (mm)", "Height_mm")),
        ("wheelbase_mm", ("wheelbase", "wheelbase_mm", "Wheelbase_mm")),
        ("doors", ("doors", "Doors")),
        ("seats", ("seats", "Seats")),
    ]
    parts = []
    for label, names in fields:
        for n in names:
            if n in row and str(row[n]).strip():
                parts.append(f"{label}={str(row[n]).strip()}")
                break
    return "; ".join(parts)


def rule_label(raw_body: str, length_mm: Optional[float]) -> Optional[str]:
    rb = (raw_body or "").lower()
    L = None
    try:
        L = float(length_mm) if length_mm is not None else None
    except Exception:
        L = None
    if any(k in rb for k in ("sportback", "fastback", "gran coupe", "gran-coupe", "grancoupe")):
        if L is not None and L >= 4850:
            return "executive hatchback (gran coupe style)"
        return "gran coupe"
    if "estate" in rb or "touring" in rb:
        return "estate"
    if "convertible" in rb or "cabrio" in rb or "roadster" in rb:
        return "convertible"
    if "mpv" in rb:
        return "MPV"
    if "coupe" in rb:
        return "coupe"
    if "suv" in rb or "crossover" in rb:
        if L is not None and L >= 4700:
            return "large SUV"
        return "crossover SUV"
    if "saloon" in rb or "sedan" in rb:
        if L is None:
            return "small saloon"
        return "executive saloon" if L >= 4700 else "small saloon"
    if "hatch" in rb:
        if L is None:
            return "medium hatchback"
        return "medium hatchback" if L >= 4300 else "small hatchback"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer canonical body type (family-level LLM + rules) and write body_type_inferred column")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--make-col", default="Make")
    ap.add_argument("--model-col", default="Model")
    ap.add_argument("--years-col", default="Series (production years start-end)")
    ap.add_argument("--body-col", default="body type")
    ap.add_argument("--max-workers", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_excel(args.input, dtype=str)
    if "body_type_inferred" not in df.columns:
        df["body_type_inferred"] = ""

    # Prepare length if available
    length_col = None
    for cand in ("Average of Length (mm)", "Length_mm", "length_mm", "average of length"):
        if cand in df.columns:
            length_col = cand
            break

    def family_id(row: pd.Series) -> str:
        return "|".join(str(x).strip().lower() for x in (row.get(args.make_col, ""), row.get(args.model_col, ""), row.get(args.years_col, "")))

    df["__family_id"] = df.apply(family_id, axis=1)

    # Rule inference per family
    fam_to_rule: Dict[str, str] = {}
    fam_to_need_llm: Dict[str, Dict[str, Any]] = {}
    for fid, grp in df.groupby("__family_id", dropna=False):
        # choose first non-empty raw body
        raw_bodies = [str(v) for v in grp.get(args.body_col, []).tolist() if str(v).strip()]
        raw_body = raw_bodies[0] if raw_bodies else ""
        L = None
        if length_col is not None:
            try:
                L = pd.to_numeric(grp[length_col], errors="coerce").dropna().astype(float).median()
            except Exception:
                L = None
        rlabel = rule_label(raw_body, L)
        fam_to_rule[fid] = rlabel or ""
        # Flag ambiguous: no rule label, or generic vs long length
        ambiguous = (not rlabel) or ("hatchback" in (rlabel or "") and (L is not None and L >= 4850)) or (raw_body.lower() in ("other", "unknown", ""))
        if ambiguous:
            # Build minimal dims context from a representative row
            rep = grp.iloc[0]
            fam_to_need_llm[fid] = {
                "make": str(rep.get(args.make_col, "")).strip(),
                "model": str(rep.get(args.model_col, "")).strip(),
                "years": str(rep.get(args.years_col, "")).strip(),
                "raw_body": raw_body,
                "dims": dims_context_from_row(rep),
            }

    # LLM for ambiguous families with concurrency
    llm_results: Dict[str, str] = {}
    if fam_to_need_llm:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futures = {}
            for fid, meta in fam_to_need_llm.items():
                fut = ex.submit(call_llm, meta["make"], meta["model"], meta["years"], meta["raw_body"], meta["dims"])
                futures[fut] = fid
            done = 0
            total = len(futures)
            start = time.time()
            for fut in as_completed(futures):
                fid = futures[fut]
                try:
                    val = fut.result()
                except Exception:
                    val = None
                if not val:
                    # fallback to rule if we had one, else medium hatchback
                    val = fam_to_rule.get(fid) or "medium hatchback"
                llm_results[fid] = val
                done += 1
                if done == 1 or done % 25 == 0:
                    rate = done / max(time.time() - start, 1e-6)
                    eta = (total - done) / rate
                    print(f"[llm-fam] {done}/{total} | {rate:.1f}/s | ETA ~{eta/60:.1f}m", flush=True)

    # Compose final family label: prefer llm_results then rule
    def fam_label(fid: str) -> str:
        if fid in llm_results:
            return llm_results[fid]
        return fam_to_rule.get(fid) or "medium hatchback"

    df["body_type_inferred"] = df["__family_id"].map(fam_label)
    df.drop(columns=["__family_id"], inplace=True)
    df.to_excel(args.output, index=False)
    print(f"Saved with 'body_type_inferred': {args.output}")


if __name__ == "__main__":
    main()


