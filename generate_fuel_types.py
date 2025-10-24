import pandas as pd
import openai
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def create_fuel_type_prompt(car_data: dict) -> str:
    """Create a minimal prompt for fuel type classification"""
    make = car_data.get('Make', '')
    model = car_data.get('Model', '')
    engine_type = car_data.get('engine type', '')
    
    return f"Classify this car's fuel type as ONLY one of: petrol, diesel, mild hybrid, plug in hybrid, electric. Car: {make} {model} {engine_type}"

def generate_fuel_type(client, car_data: dict, model: str = "gpt-4o-mini") -> str:
    """Generate fuel type for a single car"""
    try:
        prompt = create_fuel_type_prompt(car_data)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,  # Very minimal tokens needed
            temperature=0.1
        )
        
        fuel_type = response.choices[0].message.content.strip().lower()
        
        # Validate and clean the response
        valid_types = ['petrol', 'diesel', 'mild hybrid', 'plug in hybrid', 'electric']
        for valid_type in valid_types:
            if valid_type in fuel_type:
                return valid_type
        
        # Fallback based on engine type
        engine_type = car_data.get('engine type', '').lower()
        if 'electric' in engine_type or 'ev' in engine_type:
            return 'electric'
        elif 'hybrid' in engine_type and 'plug' in engine_type:
            return 'plug in hybrid'
        elif 'hybrid' in engine_type:
            return 'mild hybrid'
        elif 'diesel' in engine_type:
            return 'diesel'
        else:
            return 'petrol'
            
    except Exception as e:
        print(f"Error generating fuel type for {car_data.get('Make', 'Unknown')} {car_data.get('Model', 'Unknown')}: {e}")
        # Fallback logic
        engine_type = car_data.get('engine type', '').lower()
        if 'electric' in engine_type or 'ev' in engine_type:
            return 'electric'
        elif 'hybrid' in engine_type and 'plug' in engine_type:
            return 'plug in hybrid'
        elif 'hybrid' in engine_type:
            return 'mild hybrid'
        elif 'diesel' in engine_type:
            return 'diesel'
        else:
            return 'petrol'

def process_batch_fast(client, batch_data: list, model: str = "gpt-4o-mini") -> list:
    """Process a batch of cars for fuel type classification"""
    fuel_types = []
    
    for car_data in batch_data:
        fuel_type = generate_fuel_type(client, car_data, model)
        fuel_types.append(fuel_type)
        time.sleep(0.1)  # Minimal delay to avoid rate limits
    
    return fuel_types

def main():
    parser = argparse.ArgumentParser(description='Generate fuel types for car database')
    parser.add_argument('--input', default='db2_clean_embedded_with_display_names_simple.xlsx', help='Input Excel file path')
    parser.add_argument('--output', default='db2_clean_embedded_with_fuel_types.xlsx', help='Output Excel file path')
    parser.add_argument('--sheet', default='Sheet1', help='Excel sheet name to read from')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing (larger for speed)')
    parser.add_argument('--resume', action='store_true', help='Resume from where we left off')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input} (sheet: {args.sheet})...")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"Loaded {len(df)} cars")
    
    # Check if we're resuming
    if args.resume and os.path.exists(args.output):
        existing_df = pd.read_excel(args.output, sheet_name=args.sheet)
        if 'fuel_type' in existing_df.columns:
            print(f"Found existing output with {len(existing_df)} cars. Resuming from index {len(existing_df)}")
            start_idx = len(existing_df)
        else:
            start_idx = 0
    else:
        start_idx = 0
    
    # Initialize fuel type column
    if 'fuel_type' not in df.columns:
        df['fuel_type'] = ''
    
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
            fuel_types = process_batch_fast(client, batch_data, args.model)
            
            # Update the dataframe
            for j, fuel_type in enumerate(fuel_types):
                df.loc[i + j, 'fuel_type'] = fuel_type
            
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
    print("\nSample fuel type classifications:")
    sample_df = df[df['fuel_type'] != ''].head(10)
    for _, row in sample_df.iterrows():
        print(f"- {row['Make']} {row['Model']} → {row['fuel_type']}")
    
    # Show distribution
    print(f"\nFuel type distribution:")
    fuel_counts = df['fuel_type'].value_counts()
    print(fuel_counts)

if __name__ == "__main__":
    main()
