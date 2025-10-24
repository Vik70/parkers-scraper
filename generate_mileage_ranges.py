import pandas as pd
import openai
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm
import argparse
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def create_mileage_prompt(car_data: dict) -> str:
    """Create a prompt for mileage range generation"""
    make = car_data.get('Make', '')
    model = car_data.get('Model', '')
    variant = car_data.get('Series (production years start-end)', '')
    body = car_data.get('Real Body Type', '')
    year_start = car_data.get('Year start', '')
    year_end = car_data.get('Year end', '')
    engine_type = car_data.get('engine type', '')
    fuel_type = car_data.get('fuel_type', '')
    power_bhp = car_data.get('Power (bhp)', '')
    transmission = car_data.get('Transmission', '')
    
    # Calculate approximate age
    current_year = 2024
    if year_end and str(year_end).isdigit():
        age = current_year - int(year_end)
    elif year_start and str(year_start).isdigit():
        age = current_year - int(year_start)
    else:
        age = "unknown"
    
    prompt = f"""You are a Suggested Mileage Range Generator.

Your job is to process this car and determine a realistic mileage range that represents the miles at which a buyer can purchase this car with reasonable confidence in its reliability and market value.

Rules:
- Use your automotive knowledge first
- Consider brand reliability, engine issues, market expectations
- Assess based on make, model, engine type, age, fuel type, power, transmission
- Produce the mileage band that a typical UK buyer would consider "safe" for this specific car

Car Data:
- Make: {make}
- Model: {model}
- Variant: {variant}
- Body: {body}
- Year: {year_start}-{year_end} (Age: {age} years)
- Engine: {engine_type}
- Fuel Type: {fuel_type}
- Power: {power_bhp} bhp
- Transmission: {transmission}

Output format: Return only the suggested mileage range in format "x–y miles" (e.g., "40k–90k miles", "0–70k miles"). Round to nearest 10k. No commentary."""

    return prompt

def generate_mileage_range(client, car_data: dict, model: str = "gpt-4o-mini") -> str:
    """Generate mileage range for a single car"""
    try:
        prompt = create_mileage_prompt(car_data)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,  # Very minimal tokens needed
            temperature=0.1
        )
        
        mileage_range = response.choices[0].message.content.strip()
        
        # Clean and validate the response
        mileage_range = re.sub(r'[^\d\-\–k\s]', '', mileage_range).strip()
        
        # Validate format (should contain numbers, dash, and "miles")
        if re.search(r'\d+.*\d+.*miles', mileage_range.lower()):
            return mileage_range
        else:
            # Fallback based on fuel type and make
            fuel_type = car_data.get('fuel_type', '').lower()
            make = car_data.get('Make', '').lower()
            
            if 'electric' in fuel_type or 'hybrid' in fuel_type:
                return "0–80k miles"
            elif 'toyota' in make or 'lexus' in make or 'honda' in make:
                return "0–100k miles"
            elif 'bmw' in make or 'audi' in make or 'mercedes' in make:
                return "40k–100k miles"
            elif 'diesel' in fuel_type:
                return "40k–100k miles"
            else:
                return "0–80k miles"
                
    except Exception as e:
        print(f"Error generating mileage range for {car_data.get('Make', 'Unknown')} {car_data.get('Model', 'Unknown')}: {e}")
        # Fallback logic
        fuel_type = car_data.get('fuel_type', '').lower()
        make = car_data.get('Make', '').lower()
        
        if 'electric' in fuel_type or 'hybrid' in fuel_type:
            return "0–80k miles"
        elif 'toyota' in make or 'lexus' in make or 'honda' in make:
            return "0–100k miles"
        elif 'bmw' in make or 'audi' in make or 'mercedes' in make:
            return "40k–100k miles"
        elif 'diesel' in fuel_type:
            return "40k–100k miles"
        else:
            return "0–80k miles"

def process_batch_fast(client, batch_data: list, model: str = "gpt-4o-mini") -> list:
    """Process a batch of cars for mileage range generation"""
    mileage_ranges = []
    
    for car_data in batch_data:
        mileage_range = generate_mileage_range(client, car_data, model)
        mileage_ranges.append(mileage_range)
        time.sleep(0.1)  # Minimal delay to avoid rate limits
    
    return mileage_ranges

def main():
    parser = argparse.ArgumentParser(description='Generate suggested mileage ranges for car database')
    parser.add_argument('--input', default='db2_clean_embedded_with_fuel_types.xlsx', help='Input Excel file path')
    parser.add_argument('--output', default='db2_clean_embedded_with_mileage_ranges.xlsx', help='Output Excel file path')
    parser.add_argument('--sheet', default='Sheet1', help='Excel sheet name to read from')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--resume', action='store_true', help='Resume from where we left off')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input} (sheet: {args.sheet})...")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"Loaded {len(df)} cars")
    
    # Check if we're resuming
    if args.resume and os.path.exists(args.output):
        existing_df = pd.read_excel(args.output, sheet_name=args.sheet)
        if 'suggested_mileage_range' in existing_df.columns:
            print(f"Found existing output with {len(existing_df)} cars. Resuming from index {len(existing_df)}")
            start_idx = len(existing_df)
        else:
            start_idx = 0
    else:
        start_idx = 0
    
    # Initialize mileage range column
    if 'suggested_mileage_range' not in df.columns:
        df['suggested_mileage_range'] = ''
    
    total_cars = len(df)
    batch_size = args.batch_size
    processed = start_idx
    
    print(f"Processing {total_cars} cars in batches of {batch_size}...")
    print(f"Starting from index {start_idx}")
    
    # Process in batches
    for i in tqdm(range(start_idx, total_cars, batch_size), desc="Processing batches"):
        end_idx = min(i + batch_size, total_cars)
        batch_data = df.iloc[i:end_idx].to_dict(orient='records')
        
        try:
            mileage_ranges = process_batch_fast(client, batch_data, args.model)
            
            # Update the dataframe
            for j, mileage_range in enumerate(mileage_ranges):
                df.loc[i + j, 'suggested_mileage_range'] = mileage_range
            
            processed += len(batch_data)
            
            # Save progress every 1000 cars
            if processed % 1000 == 0:
                print(f"\nSaving progress... {processed}/{total_cars} cars processed")
                df.to_excel(args.output, index=False, sheet_name=args.sheet)
                
        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}")
            continue
    
    # Final save
    print(f"\nSaving final results to {args.output}...")
    df.to_excel(args.output, index=False, sheet_name=args.sheet)
    
    print(f"Completed! Processed {processed} cars")
    print(f"Results saved to {args.output}")
    
    # Show sample results
    print("\nSample mileage range classifications:")
    sample_df = df[df['suggested_mileage_range'] != ''].head(10)
    for _, row in sample_df.iterrows():
        print(f"- {row['Make']} {row['Model']} ({row['fuel_type']}) → {row['suggested_mileage_range']}")
    
    # Show distribution
    print(f"\nMileage range distribution:")
    mileage_counts = df['suggested_mileage_range'].value_counts().head(10)
    print(mileage_counts)

if __name__ == "__main__":
    main()
