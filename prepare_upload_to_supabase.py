import pandas as pd
import openai
from dotenv import load_dotenv
import os
import numpy as np
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def create_spec_text(car_data: dict) -> str:
    """Create a specs text string from car data"""
    specs = []
    
    if car_data.get('Power (bhp)'):
        specs.append(f"Power: {car_data['Power (bhp)']} bhp")
    if car_data.get('Transmission'):
        specs.append(f"Transmission: {car_data['Transmission']}")
    if car_data.get('Doors'):
        specs.append(f"Doors: {car_data['Doors']}")
    if car_data.get('Seats'):
        specs.append(f"Seats: {car_data['Seats']}")
    if car_data.get('engine type'):
        specs.append(f"Engine: {car_data['engine type']}")
    if car_data.get('fuel_type'):
        specs.append(f"Fuel: {car_data['fuel_type']}")
    if car_data.get('Real Body Type'):
        specs.append(f"Body: {car_data['Real Body Type']}")
    
    return ", ".join(specs)

def embed_text(client, text: str, model: str = "text-embedding-3-small") -> list:
    """Embed text using OpenAI embeddings"""
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error embedding text: {e}")
        # Return zero vector of correct dimension
        dimension = 1536 if model == "text-embedding-3-small" else 3072
        return [0.0] * dimension

def embed_batch(client, texts: list, model: str = "text-embedding-3-small") -> list:
    """Embed a batch of texts"""
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error in batch embedding: {e}")
        dimension = 1536 if model == "text-embedding-3-small" else 3072
        return [[0.0] * dimension for _ in texts]

def main():
    input_file = 'db2_clean_embedded_with_mileage_ranges.xlsx'
    output_csv = 'cars_for_supabase_upload.csv'
    
    print(f"Loading data from {input_file}...")
    df = pd.read_excel(input_file)
    print(f"Loaded {len(df)} cars")
    
    # Remove the suggested_mileage_range column as requested
    if 'suggested_mileage_range' in df.columns:
        df = df.drop('suggested_mileage_range', axis=1)
    
    # Create specs text for each car
    print("Creating specs text...")
    df['specs_text'] = df.apply(create_spec_text, axis=1)
    
    # Embed essence_main and specs_text
    print("Embedding texts...")
    
    # Prepare texts for embedding
    essence_texts = df['essence_main'].fillna('').tolist()
    specs_texts = df['specs_text'].fillna('').tolist()
    
    essence_embeddings = []
    specs_embeddings = []
    
    batch_size = 100
    
    # Embed essence_main in batches
    print("Embedding essence_main...")
    for i in tqdm(range(0, len(essence_texts), batch_size), desc="Embedding essence"):
        batch = essence_texts[i:i+batch_size]
        batch_embeddings = embed_batch(client, batch)
        essence_embeddings.extend(batch_embeddings)
        time.sleep(0.1)  # Rate limiting
    
    # Embed specs_text in batches
    print("Embedding specs_text...")
    for i in tqdm(range(0, len(specs_texts), batch_size), desc="Embedding specs"):
        batch = specs_texts[i:i+batch_size]
        batch_embeddings = embed_batch(client, batch)
        specs_embeddings.extend(batch_embeddings)
        time.sleep(0.1)  # Rate limiting
    
    # Add embeddings to dataframe
    df['vector_main'] = essence_embeddings
    df['vector_specs'] = specs_embeddings
    
    # Convert embeddings to string format for CSV
    df['vector_main_text'] = df['vector_main'].apply(lambda x: str(x))
    df['vector_specs_text'] = df['vector_specs'].apply(lambda x: str(x))
    
    # Map column names to match your Supabase schema
    column_mapping = {
        'Make': 'make',
        'Model': 'model',
        'Series (production years start-end)': 'series_production_years',
        'Year start': 'year_start',
        'Year end': 'year_end',
        'Real Body Type': 'real_body_type',
        'engine type': 'engine_type',
        'Power (bhp)': 'power_bhp',
        'Transmission': 'transmission',
        'Doors': 'doors',
        'Seats': 'seats',
        'Min. of used price low helper': 'used_price_low_min',
        'Max. of used price high helper': 'used_price_high_max',
        'Min. of mpg low helper': 'mpg_low_min',
        'Min. of Insurance group': 'insurance_group_min',
        'Min. of 0-60 mph (secs)': 'zero_to_sixty_secs_min',
        'Average of Length (mm)': 'length_mm_avg',
        'Average of Width (mm)': 'width_mm_avg',
        'Average of Height (mm)': 'height_mm_avg',
        'Average of Luggage Capacity (litres)': 'luggage_capacity_litres_avg',
        'essence_main': 'essence_main',
        'Image URL': 'image_urls',
        'display_name': 'display_name',
        'fuel_type': 'fuel_type'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need, excluding the vector arrays (keep only text versions)
    columns_to_keep = [
        'make', 'model', 'series_production_years', 'year_start', 'year_end',
        'real_body_type', 'engine_type', 'power_bhp', 'transmission', 'doors', 'seats',
        'used_price_low_min', 'used_price_high_max', 'mpg_low_min', 'insurance_group_min',
        'zero_to_sixty_secs_min', 'length_mm_avg', 'width_mm_avg', 'height_mm_avg',
        'luggage_capacity_litres_avg', 'essence_main', 'image_urls', 'display_name',
        'fuel_type', 'specs_text', 'vector_main_text', 'vector_specs_text'
    ]
    
    df_output = df[columns_to_keep].copy()
    
    # Save to CSV
    print(f"Saving to {output_csv}...")
    df_output.to_csv(output_csv, index=False)
    
    print(f"Completed! Saved {len(df_output)} cars to {output_csv}")
    print(f"Columns: {list(df_output.columns)}")
    
    # Show sample
    print("\nSample row:")
    print(df_output.head(1).to_dict('records')[0])

if __name__ == "__main__":
    main()
