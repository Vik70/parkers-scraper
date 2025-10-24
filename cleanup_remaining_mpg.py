#!/usr/bin/env python3
"""
Clean up remaining zero MPG values using LLM inference.
"""

import pandas as pd
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def call_llm_for_mpg(meta: dict) -> dict:
    """Call LLM to infer MPG for a single car."""
    client = OpenAI()
    
    prompt = (
        "Infer realistic UK MPG (miles per gallon) for this car. Return JSON only with these exact keys:\n"
        "{\n"
        '  "mpg_low_min": number (typical minimum MPG for this car),\n'
        '  "confidence": "high"|"medium"|"low",\n'
        '  "notes": "brief explanation"\n'
        "}\n\n"
        "Rules:\n"
        "- Use current UK market context and typical values for this car type\n"
        "- MPG: realistic fuel economy for the engine/transmission combination\n"
        "- Be conservative but realistic - not extreme values\n"
        "- If series spans many years, use representative values for the family\n"
        "- Consider engine size, power, weight, and transmission type\n\n"
        f"Make: {meta.get('Make','').strip()}\n"
        f"Model: {meta.get('Model','').strip()}\n"
        f"Series/Years: {meta.get('Series','').strip()} | start={meta.get('YearStart','')} end={meta.get('YearEnd','')}\n"
        f"Body: {meta.get('Body','')} | Engine: {meta.get('Engine','')} | Power(bhp): {meta.get('Power','')} | Trans: {meta.get('Trans','')}\n"
        f"Doors: {meta.get('Doors','')} | Seats: {meta.get('Seats','')}\n"
        f"Current MPG: {meta.get('CurrentMPG','')} (needs to be filled)"
    )
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        content = resp.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
        
        return {
            "mpg_low_min": float(result.get("mpg_low_min", 0)),
            "confidence": result.get("confidence", "low"),
            "notes": result.get("notes", "")
        }
        
    except Exception as e:
        print(f"Error processing {meta.get('Make')} {meta.get('Model')}: {e}")
        return {
            "mpg_low_min": 0,
            "confidence": "low", 
            "notes": f"Error: {str(e)}"
        }

def main():
    # Read the data
    print("Loading data...")
    df = pd.read_excel('database_2.0_specs_final.xlsx')
    
    print(f"Total cars: {len(df)}")
    
    # Find cars with zero MPG
    zero_mpg = df[df["Min. of mpg low helper"] == 0]
    print(f"Cars with zero MPG: {len(zero_mpg)}")
    
    if len(zero_mpg) == 0:
        print("No cars with zero MPG found!")
        return
    
    # Process each car with zero MPG
    print("Processing cars with zero MPG...")
    results = []
    
    for idx, (_, row) in enumerate(zero_mpg.iterrows()):
        if idx % 50 == 0:
            print(f"Processing {idx + 1}/{len(zero_mpg)}...")
        
        # Build metadata
        meta = {
            "Make": str(row.get("Make", "")),
            "Model": str(row.get("Model", "")),
            "Series": str(row.get("Series (production years start-end)", "")),
            "YearStart": str(row.get("Year start", "")),
            "YearEnd": str(row.get("Year end", "")),
            "Body": str(row.get("Real Body Type", "")),
            "Engine": str(row.get("engine type", "")),
            "Power": str(row.get("Power (bhp)", "")),
            "Trans": str(row.get("Transmission", "")),
            "Doors": str(row.get("Doors", "")),
            "Seats": str(row.get("Seats", "")),
            "CurrentMPG": str(row.get("Min. of mpg low helper", ""))
        }
        
        # Call LLM
        result = call_llm_for_mpg(meta)
        result["index"] = idx
        results.append(result)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Apply results
    print("Applying results...")
    for i, result in enumerate(results):
        if result["mpg_low_min"] > 0:  # Only update if we got a valid result
            zero_mpg_idx = zero_mpg.index[i]
            df.loc[zero_mpg_idx, "Min. of mpg low helper"] = result["mpg_low_min"]
    
    # Save results
    output_file = "database_2.0_specs_clean.xlsx"
    print(f"Saving to {output_file}...")
    df.to_excel(output_file, index=False)
    
    # Final stats
    final_zero_mpg = (df["Min. of mpg low helper"] == 0).sum()
    print(f"\nFinal results:")
    print(f"Total cars: {len(df)}")
    print(f"Remaining zero MPG: {final_zero_mpg}")
    print(f"MPG range: {df['Min. of mpg low helper'].min()} - {df['Min. of mpg low helper'].max()}")
    
    print("Done!")

if __name__ == "__main__":
    main()
