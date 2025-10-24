#!/usr/bin/env python3
"""
Generate Display Names for Car Database

This script reads the car database and generates display names for every car
using GPT-4o with the provided prompt. It processes all cars and adds a new
'display_name' column to the output.
"""

import pandas as pd
import openai
import os
import time
from typing import List, Dict, Any
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def setup_openai():
    """Setup OpenAI client with API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = openai.OpenAI(api_key=api_key)
    return client

def create_car_prompt(car_data: Dict[str, Any]) -> str:
    """Create the prompt for a single car based on the provided template"""
    
    # Extract key information
    make = car_data.get('Make', '')
    model = car_data.get('Model', '')
    series = car_data.get('Series (production years start-end)', '')
    year_start = car_data.get('Year start', '')
    year_end = car_data.get('Year end', '')
    real_body_type = car_data.get('Real Body Type', '')
    engine_type = car_data.get('engine type', '')
    power_bhp = car_data.get('Power (bhp)', '')
    transmission = car_data.get('Transmission', '')
    doors = car_data.get('Doors', '')
    seats = car_data.get('Seats', '')
    
    # Create the car description for the prompt
    car_info = f"""
Car Data:
- Make: {make}
- Model: {model}
- Series: {series}
- Production Years: {year_start}-{year_end}
- Body Type: {real_body_type}
- Engine: {engine_type}
- Power: {power_bhp} bhp
- Transmission: {transmission}
- Doors: {doors}
- Seats: {seats}
"""

    prompt = f"""Car Display Name Generator

You are a Car Display Name Generator.
Your job is to take each row in our car database and generate a clear, short, and accurate Display Name that will be shown to users on our site.

⸻
Rules

1. Output format

Always output exactly:
<Make> <Model/Series/Badge/Variant>
Examples:
	•	BMW M3 Saloon
	•	BMW 325d Gran Turismo
	•	Mercedes-AMG GT 4-Door Coupe
	•	Audi RS 3 Sportback
	•	Porsche Taycan Turbo S Sport Turismo
	•	Tesla Model S Plaid

⸻

2. Brand names
	•	Use correct manufacturer naming: Mercedes-Benz, BMW, Audi, Porsche, Volkswagen, Tesla, Škoda, SEAT, Cupra, Land Rover, Aston Martin, Jaguar, Lexus, etc.
	•	When applicable, use performance sub-brands: Mercedes-AMG, BMW M, Audi RS/S, Cupra, Škoda vRS, Ford ST/RS, NISMO, Quadrifoglio, etc.

⸻

3. Model/variant/badge selection
	•	If the car belongs to a performance family (BMW M, AMG, RS, Porsche Turbo/GT, Cupra, vRS, NISMO, Quadrifoglio, GTI, R, ST, etc.), use that.
	•	Otherwise, generate the correct numeric badge (e.g., 320d, 325d, 330d, 330i, 40 TDI, 220d, etc.) based on engine size, fuel, power_bhp, and year.
	•	Use your own automotive knowledge first; if uncertain, briefly check the web.
	•	Examples:
	•	BMW 3 Series 2.0d ~188 bhp ⇒ BMW 320d Saloon
	•	BMW 3 Series 2.0d ~218 bhp ⇒ BMW 325d Saloon
	•	Audi A4 diesel ~190 bhp (2019) ⇒ Audi A4 40 TDI Saloon
	•	C-Class 4.0 V8 510 bhp (2016) ⇒ Mercedes-AMG C 63 S Saloon

⸻

4. Body/model terms
	•	Always use the official model name as sold in the UK.
	•	Examples:
	•	BMW 6 Series Gran Coupe
	•	BMW 3 Series Gran Turismo
	•	Audi A5 Sportback
	•	Audi A4 Cabriolet
	•	Porsche Taycan Sport Turismo
	•	Do not force-fit into a short generic list (like just "Saloon"/"Estate"/"Coupe"); if the true model name includes Gran Coupe, Gran Turismo, Sportback, Cabriolet, 4-Door Coupe, Sport Turismo, etc., keep that wording exactly.

⸻

5. Keep it clean
	•	Do not include engine size, bhp, drivetrain, gearbox, or insurance group.
	•	Only include trim tokens if they are part of the official name (e.g., Competition, Plaid, Turbo S, Black Series, CSL, Performance, Long Range, NISMO, Quadrifoglio, GTI, vRS, Cupra, etc.).

⸻

6. If unsure
	•	Deduce from your own knowledge + the row's data first.
	•	If still uncertain, check the web briefly.
	•	Always return the closest correct official UK-market name.
	•	Never leave blank.

⸻

Output contract
	•	For each row: output only the final Display Name string.
	•	One line per row.
	•	No commentary, no JSON, just the clean name.

⸻

✅ This ensures your Display Name column is clean and user-friendly.
Examples you'd get back:
	•	BMW 330d Saloon
	•	BMW 640d Gran Coupe
	•	Audi RS 6 Avant
	•	Mercedes-AMG E 63 S Estate
	•	Porsche 911 GT3 RS
	•	Tesla Model X Plaid

{car_info}

Generate the display name for this car:"""

    return prompt

def generate_display_name(client: openai.OpenAI, car_data: Dict[str, Any], model: str = "gpt-4o") -> str:
    """Generate display name for a single car using GPT"""
    
    prompt = create_car_prompt(car_data)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        display_name = response.choices[0].message.content.strip()
        
        # Clean up the response - remove any extra text or formatting
        display_name = display_name.replace('"', '').replace("'", "").strip()
        
        return display_name
        
    except Exception as e:
        print(f"Error generating display name for {car_data.get('Make', 'Unknown')} {car_data.get('Model', 'Unknown')}: {e}")
        return f"{car_data.get('Make', 'Unknown')} {car_data.get('Model', 'Unknown')}"

def process_batch(client: openai.OpenAI, batch: List[Dict[str, Any]], model: str = "gpt-4o") -> List[str]:
    """Process a batch of cars and return their display names"""
    display_names = []
    
    for car_data in batch:
        display_name = generate_display_name(client, car_data, model)
        display_names.append(display_name)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return display_names

def main():
    parser = argparse.ArgumentParser(description='Generate display names for car database')
    parser.add_argument('--input', required=True, help='Input Excel file path')
    parser.add_argument('--output', required=True, help='Output Excel file path')
    parser.add_argument('--sheet', default='Sheet1', help='Excel sheet name to read from')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of worker threads')
    parser.add_argument('--resume', action='store_true', help='Resume from where we left off')
    parser.add_argument('--start-index', type=int, default=0, help='Start processing from this index')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input} (sheet: {args.sheet})...")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"Loaded {len(df)} cars")
    
    # Check if we're resuming
    if args.resume and os.path.exists(args.output):
        existing_df = pd.read_excel(args.output, sheet_name=args.sheet)
        if 'display_name' in existing_df.columns:
            print(f"Found existing output with {len(existing_df)} cars. Resuming from index {len(existing_df)}")
            start_idx = len(existing_df)
        else:
            start_idx = 0
    else:
        start_idx = args.start_index
    
    # Initialize display names column
    if 'display_name' not in df.columns:
        df['display_name'] = ''
    
    # Setup OpenAI client
    client = setup_openai()
    
    # Process cars in batches
    total_cars = len(df)
    processed = 0
    
    print(f"Processing {total_cars - start_idx} cars starting from index {start_idx}...")
    
    # Create batches
    batch_size = args.batch_size
    batches = []
    
    for i in range(start_idx, total_cars, batch_size):
        end_idx = min(i + batch_size, total_cars)
        batch_data = df.iloc[i:end_idx].to_dict('records')
        batches.append((i, batch_data))
    
    # Process batches with progress bar
    with tqdm(total=len(batches), desc="Processing batches") as pbar:
        for batch_start, batch_data in batches:
            try:
                display_names = process_batch(client, batch_data, args.model)
                
                # Update the dataframe
                for j, display_name in enumerate(display_names):
                    df.loc[batch_start + j, 'display_name'] = display_name
                
                processed += len(batch_data)
                pbar.update(1)
                
                # Save progress every 100 cars
                if processed % 100 == 0:
                    print(f"\nSaving progress... {processed}/{total_cars} cars processed")
                    df.to_excel(args.output, index=False, sheet_name=args.sheet)
                
            except Exception as e:
                print(f"\nError processing batch starting at {batch_start}: {e}")
                continue
    
    # Final save
    print(f"\nSaving final results to {args.output}...")
    df.to_excel(args.output, index=False, sheet_name=args.sheet)
    
    print(f"Completed! Processed {processed} cars")
    print(f"Results saved to {args.output}")
    
    # Show some sample results
    print("\nSample display names:")
    sample_df = df[df['display_name'] != ''].head(10)
    for _, row in sample_df.iterrows():
        print(f"- {row['display_name']}")

if __name__ == "__main__":
    main()
