#!/usr/bin/env python3
"""
Test script to generate display names for a few cars
"""

import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_openai():
    """Setup OpenAI client with API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = openai.OpenAI(api_key=api_key)
    return client

def create_car_prompt(car_data):
    """Create the prompt for a single car"""
    
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

def generate_display_name(client, car_data):
    """Generate display name for a single car using GPT"""
    
    prompt = create_car_prompt(car_data)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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

def main():
    print("Loading data from db2_clean_embedded.xlsx...")
    df = pd.read_excel('db2_clean_embedded.xlsx')
    print(f"Loaded {len(df)} cars")
    
    # Take first 5 cars for testing
    test_df = df.head(5).copy()
    
    # Setup OpenAI client
    client = setup_openai()
    
    print("Generating display names for test cars...")
    display_names = []
    
    for i, (_, car_data) in enumerate(test_df.iterrows()):
        print(f"Processing car {i+1}/5: {car_data['Make']} {car_data['Model']}")
        display_name = generate_display_name(client, car_data)
        display_names.append(display_name)
        print(f"Generated: {display_name}")
        print()
    
    # Add display names to dataframe
    test_df['display_name'] = display_names
    
    # Save results
    test_df.to_excel('test_display_names_sample.xlsx', index=False)
    
    print("Test completed! Results saved to test_display_names_sample.xlsx")
    print("\nSample results:")
    for _, row in test_df.iterrows():
        print(f"- {row['Make']} {row['Model']} → {row['display_name']}")

if __name__ == "__main__":
    main()
