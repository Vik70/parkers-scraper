#!/usr/bin/env python3
"""
Test if the LLM call is working.
"""

import pandas as pd
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI client
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set")
    exit(1)

client = OpenAI()

# Read the data
df = pd.read_excel('database_2.0_enriched.xlsx', sheet_name='Sheet2')

# Find a car with missing values
missing_mpg = df[df["Min. of mpg low helper"] == 0]
if len(missing_mpg) == 0:
    print("No cars with missing MPG found")
    exit(1)

sample_row = missing_mpg.iloc[0]
print(f"Testing with car: {sample_row['Make']} {sample_row['Model']}")

# Build the prompt
meta = {
    "Make": str(sample_row.get("Make", "")),
    "Model": str(sample_row.get("Model", "")),
    "Series": str(sample_row.get("Series (production years start-end)", "")),
    "YearStart": str(sample_row.get("Year start", "")),
    "YearEnd": str(sample_row.get("Year end", "")),
    "Body": str(sample_row.get("Real Body Type", "")),
    "Engine": str(sample_row.get("engine type", "")),
    "Power": str(sample_row.get("Power (bhp)", "")),
    "Trans": str(sample_row.get("Transmission", "")),
    "Doors": str(sample_row.get("Doors", "")),
    "Seats": str(sample_row.get("Seats", "")),
    "CurrentMPG": str(sample_row.get("Min. of mpg low helper", "")),
    "CurrentInsurance": str(sample_row.get("Min. of Insurance group", "")),
    "Current060": str(sample_row.get("Min. of 0-60 mph (secs)", ""))
}

prompt = (
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

print(f"\nPrompt:\n{prompt}\n")

# Call LLM
try:
    print("Calling LLM...")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    
    content = resp.choices[0].message.content.strip()
    print(f"LLM Response:\n{content}\n")
    
    # Try to parse JSON
    try:
        result = json.loads(content)
        print(f"Parsed JSON: {result}")
    except json.JSONDecodeError:
        print("Failed to parse JSON")
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            print(f"Extracted JSON: {result}")
        else:
            print("No JSON found in response")
            
except Exception as e:
    print(f"Error calling LLM: {e}")
