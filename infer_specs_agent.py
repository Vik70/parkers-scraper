#!/usr/bin/env python3
"""
Agent to infer missing car specifications using LLM.
Fills in missing values for:
- Min. of mpg low helper
- Min. of Insurance group  
- Min. of 0-60 mph (secs)

Processes ALL cars in the database (34,247 total).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
from openai import OpenAI

# Global client for reuse
client = None

def get_client() -> OpenAI:
    """Get OpenAI client, creating if needed."""
    global client
    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI()
    return client

def normalize_year(value) -> str:
    """Normalize year values."""
    if pd.isna(value) or value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ["now", "present", "current"]:
        return "2024"
    return s

def build_specs_prompt(meta: Dict[str, str]) -> str:
    """Build prompt for specs inference."""
    return (
        "Infer UK car specifications for this vehicle. Return JSON only with these exact keys:\n"
        "{\n"
        '  "mpg_low_min": number (typical minimum MPG for this car),\n'
        '  "insurance_group_min": number (UK insurance group 1-50, lower=cheaper),\n'
        '  "zero_to_sixty_secs_min": number (0-60 mph time in seconds),\n'
        '  "confidence": "high"|"medium"|"low",\n'
        '  "notes": "brief explanation"\n'
        "}\n\n"
        "Rules:\n"
        "- Use current UK market context and typical values for this car type\n"
        "- MPG: realistic fuel economy for the engine/transmission combination\n"
        "- Insurance group: 1-50 scale, consider engine size, power, value, theft risk\n"
        "- 0-60: realistic acceleration time for the power/weight combination\n"
        "- Be conservative but realistic - not extreme values\n"
        "- If series spans many years, use representative values for the family\n\n"
        f"Make: {meta.get('Make','').strip()}\n"
        f"Model: {meta.get('Model','').strip()}\n"
        f"Series/Years: {meta.get('Series','').strip()} | start={meta.get('YearStart','')} end={meta.get('YearEnd','')}\n"
        f"Body: {meta.get('Body','')} | Engine: {meta.get('Engine','')} | Power(bhp): {meta.get('Power','')} | Trans: {meta.get('Trans','')}\n"
        f"Doors: {meta.get('Doors','')} | Seats: {meta.get('Seats','')}\n"
        f"Current MPG: {meta.get('CurrentMPG','')} | Current Insurance: {meta.get('CurrentInsurance','')} | Current 0-60: {meta.get('Current060','')}"
    )

def call_llm_specs_agent(meta: Dict[str, str], model: str = "gpt-4o") -> Dict:
    """Call LLM to infer specs."""
    try:
        client = get_client()
        prompt = build_specs_prompt(meta)
        
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        content = resp.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {"error": f"Could not parse JSON: {content[:200]}"}
                
    except Exception as e:
        return {"error": str(e)}

def process_car_specs(row: pd.Series, model: str = "gpt-4o") -> Dict:
    """Process a single car for specs inference."""
    # Build metadata
    meta = {
        "Make": str(row.get("Make", "")),
        "Model": str(row.get("Model", "")),
        "Series": str(row.get("Series (production years start-end)", "")),
        "YearStart": normalize_year(row.get("Year start", "")),
        "YearEnd": normalize_year(row.get("Year end", "")),
        "Body": str(row.get("Real Body Type", "")),
        "Engine": str(row.get("engine type", "")),
        "Power": str(row.get("Power (bhp)", "")),
        "Trans": str(row.get("Transmission", "")),
        "Doors": str(row.get("Doors", "")),
        "Seats": str(row.get("Seats", "")),
        "CurrentMPG": str(row.get("Min. of mpg low helper", "")),
        "CurrentInsurance": str(row.get("Min. of Insurance group", "")),
        "Current060": str(row.get("Min. of 0-60 mph (secs)", ""))
    }
    
    # Check if we need to fill any values
    needs_mpg = meta["CurrentMPG"] in ["0", "0.0", ""] or pd.isna(row.get("Min. of mpg low helper")) or row.get("Min. of mpg low helper") == 0
    needs_insurance = meta["CurrentInsurance"] in ["0", "0.0", ""] or pd.isna(row.get("Min. of Insurance group")) or row.get("Min. of Insurance group") == 0
    needs_060 = meta["Current060"] in ["0", "0.0", ""] or pd.isna(row.get("Min. of 0-60 mph (secs)")) or row.get("Min. of 0-60 mph (secs)") == 0
    
    if not (needs_mpg or needs_insurance or needs_060):
        return {
            "mpg_low_min": float(meta["CurrentMPG"]) if meta["CurrentMPG"] else None,
            "insurance_group_min": int(meta["CurrentInsurance"]) if meta["CurrentInsurance"] else None,
            "zero_to_sixty_secs_min": float(meta["Current060"]) if meta["Current060"] else None,
            "confidence": "existing",
            "notes": "Using existing values"
        }
    
    # Call LLM
    result = call_llm_specs_agent(meta, model)
    
    if "error" in result:
        return result
    
    # Validate and clean results
    try:
        mpg = result.get("mpg_low_min")
        if mpg is not None:
            mpg = float(mpg)
            if mpg <= 0 or mpg > 200:
                mpg = None
        
        insurance = result.get("insurance_group_min")
        if insurance is not None:
            insurance = int(insurance)
            if insurance < 1 or insurance > 50:
                insurance = None
        
        zero60 = result.get("zero_to_sixty_secs_min")
        if zero60 is not None:
            zero60 = float(zero60)
            if zero60 <= 0 or zero60 > 30:
                zero60 = None
        
        return {
            "mpg_low_min": mpg if needs_mpg else (float(meta["CurrentMPG"]) if meta["CurrentMPG"] else None),
            "insurance_group_min": insurance if needs_insurance else (int(meta["CurrentInsurance"]) if meta["CurrentInsurance"] else None),
            "zero_to_sixty_secs_min": zero60 if needs_060 else (float(meta["Current060"]) if meta["Current060"] else None),
            "confidence": result.get("confidence", "low"),
            "notes": result.get("notes", "")
        }
        
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid result format: {e}"}

def process_batch_specs(batch: List[Tuple[int, pd.Series]], model: str = "gpt-4o") -> List[Tuple[int, Dict]]:
    """Process a batch of cars for specs inference."""
    results = []
    for idx, row in batch:
        try:
            result = process_car_specs(row, model)
            results.append((idx, result))
        except Exception as e:
            results.append((idx, {"error": str(e)}))
    return results

def main():
    parser = argparse.ArgumentParser(description="Infer missing car specifications using LLM")
    parser.add_argument("--input", required=True, help="Input Excel file")
    parser.add_argument("--sheet", default="Sheet2", help="Sheet name to process")
    parser.add_argument("--output", required=True, help="Output Excel file")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--max-workers", type=int, default=16, help="Max concurrent workers")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all values")
    parser.add_argument("--only-missing", action="store_true", help="Only process cars with missing values")
    parser.add_argument("--dry-run", action="store_true", help="Dry run - only process first 20 cars")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"Loaded {len(df)} cars")
    
    # Check current missing values
    mpg_missing = (df["Min. of mpg low helper"] == 0).sum()
    insurance_missing = (df["Min. of Insurance group"] == 0).sum()
    zero60_missing = (df["Min. of 0-60 mph (secs)"] == 0).sum()
    
    print(f"Missing values:")
    print(f"  MPG low: {mpg_missing}")
    print(f"  Insurance group: {insurance_missing}")
    print(f"  0-60 mph: {zero60_missing}")
    
    if args.dry_run:
        print("DRY RUN: Processing only first 20 cars")
        df = df.head(20)
    
    # Prepare results
    results = {}
    
    # Create batches
    batch_size = args.batch_size
    batches = []
    for i in range(0, len(df), batch_size):
        batch = [(idx, row) for idx, row in df.iloc[i:i+batch_size].iterrows()]
        batches.append(batch)
    
    print(f"Processing {len(batches)} batches with {args.max_workers} workers...")
    
    start_time = time.time()
    processed = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_batch_specs, batch, args.model): batch 
            for batch in batches
        }
        
        # Process completed batches
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                for idx, result in batch_results:
                    results[idx] = result
                
                processed += len(batch_results)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(df) - processed) / rate if rate > 0 else 0
                
                print(f"[specs] {processed}/{len(df)} | {rate:.1f}/s | ETA ~{eta/60:.1f}m")
                
            except Exception as e:
                print(f"Batch failed: {e}")
    
    # Apply results to dataframe
    print("Applying results to dataframe...")
    
    for idx, result in results.items():
        if "error" not in result:
            if result.get("mpg_low_min") is not None:
                df.loc[idx, "Min. of mpg low helper"] = result["mpg_low_min"]
            if result.get("insurance_group_min") is not None:
                df.loc[idx, "Min. of Insurance group"] = result["insurance_group_min"]
            if result.get("zero_to_sixty_secs_min") is not None:
                df.loc[idx, "Min. of 0-60 mph (secs)"] = result["zero_to_sixty_secs_min"]
    
    # Save results
    print(f"Saving to {args.output}...")
    df.to_excel(args.output, index=False)
    
    # Final stats
    final_mpg_missing = (df["Min. of mpg low helper"] == 0).sum()
    final_insurance_missing = (df["Min. of Insurance group"] == 0).sum()
    final_zero60_missing = (df["Min. of 0-60 mph (secs)"] == 0).sum()
    
    print(f"Final missing values:")
    print(f"  MPG low: {final_mpg_missing} (was {mpg_missing})")
    print(f"  Insurance group: {final_insurance_missing} (was {insurance_missing})")
    print(f"  0-60 mph: {final_zero60_missing} (was {zero60_missing})")
    
    print("Done!")

if __name__ == "__main__":
    main()
