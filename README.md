# Parkers Car Database Scraper & AI Enhancement

This repository contains scripts for processing and enhancing a comprehensive UK car database with AI-generated content using OpenAI's GPT models.

## 🚗 What This Repository Does

This project processes a database of **34,247 UK cars** and enhances it with AI-generated content:

1. **Display Names**: Creates automotive display names using GPT-4o-mini
2. **Fuel Types**: Categorizes cars by fuel type (petrol, diesel, hybrid, electric)
3. **Mileage Ranges**: Generates suggested mileage ranges for UK car buyers

## 📁 Key Files

### Main Processing Scripts
- `generate_display_names_simple.py` - Creates automotive display names
- `generate_fuel_types.py` - Categorizes fuel types
- `generate_mileage_ranges.py` - Generates mileage ranges

### Data Files
- `db2_clean_embedded.xlsx` - Original car database (34,247 cars)
- `db2_clean_embedded_with_display_names_simple.xlsx` - With display names
- `db2_clean_embedded_with_fuel_types.xlsx` - With fuel types
- `db2_clean_embedded_with_mileage_ranges.xlsx` - Final enhanced database

## 🛠️ Setup

### Prerequisites
```bash
pip install pandas openai python-dotenv tqdm openpyxl
```

### Environment Setup
1. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 Usage

### 1. Generate Display Names
```bash
python generate_display_names_simple.py
```
- Processes all 34,247 cars
- Creates automotive display names like "Audi A6 2.0 TDI Saloon"
- Output: `db2_clean_embedded_with_display_names_simple.xlsx`

### 2. Generate Fuel Types
```bash
python generate_fuel_types.py --batch-size 100
```
- Categorizes cars into: petrol, diesel, mild hybrid, plug in hybrid, electric
- Fast processing with large batches
- Output: `db2_clean_embedded_with_fuel_types.xlsx`

### 3. Generate Mileage Ranges
```bash
python generate_mileage_ranges.py --batch-size 100
```
- Creates suggested mileage ranges for UK buyers
- Uses automotive knowledge for reliability patterns
- Output: `db2_clean_embedded_with_mileage_ranges.xlsx`

## 📊 Results

### Display Names Examples
- Abarth 500 → "Abarth 500 Esseesse"
- Audi A6 → "Audi A6 2.0 TDI Saloon"
- Volvo XC90 → "Volvo XC90 T8 Twin Engine AWD"

### Fuel Type Distribution
- **Petrol**: 18,565 cars (54.2%)
- **Diesel**: 13,247 cars (38.7%)
- **Electric**: 1,530 cars (4.5%)
- **Plug-in Hybrid**: 632 cars (1.8%)
- **Mild Hybrid**: 273 cars (0.8%)

### Mileage Range Examples
- Toyota Prius (hybrid) → "0–100k miles"
- BMW 3 Series (diesel) → "40k–100k miles"
- Ford Focus (petrol) → "0–80k miles"
- Tesla Model S (electric) → "0–80k miles"

## 💰 Cost Analysis

Using GPT-4o-mini for cost efficiency:
- **Display Names**: ~$5 (vs ~$161 with GPT-4o)
- **Fuel Types**: ~$2
- **Mileage Ranges**: ~$3
- **Total**: ~$10 for all 34,247 cars

## 🗄️ Uploading to Supabase

### Step 1: Prepare the Data
1. Use the final file: `db2_clean_embedded_with_mileage_ranges.xlsx`
2. This contains all original data plus:
   - `display_name` column
   - `fuel_type` column  
   - `suggested_mileage_range` column

### Step 2: Convert to CSV
```python
import pandas as pd
df = pd.read_excel('db2_clean_embedded_with_mileage_ranges.xlsx')
df.to_csv('cars_database_final.csv', index=False)
```

### Step 3: Create Supabase Table
```sql
-- Create the cars table
CREATE TABLE cars (
    id SERIAL PRIMARY KEY,
    make VARCHAR(100),
    model VARCHAR(100),
    series_production_years VARCHAR(100),
    year_start INTEGER,
    year_end INTEGER,
    real_body_type VARCHAR(100),
    engine_type VARCHAR(100),
    power_bhp INTEGER,
    transmission VARCHAR(100),
    doors INTEGER,
    seats INTEGER,
    min_used_price_low DECIMAL(10,2),
    max_used_price_high DECIMAL(10,2),
    min_mpg_low DECIMAL(5,2),
    min_insurance_group INTEGER,
    min_0_60_secs DECIMAL(4,2),
    avg_length_mm INTEGER,
    avg_width_mm INTEGER,
    avg_height_mm INTEGER,
    avg_luggage_capacity_litres INTEGER,
    essence_main TEXT,
    image_url TEXT,
    display_name VARCHAR(200),
    fuel_type VARCHAR(50),
    suggested_mileage_range VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Step 4: Upload Data
1. Go to Supabase Dashboard → Table Editor
2. Click "Import data from CSV"
3. Upload `cars_database_final.csv`
4. Map columns to the table schema
5. Click "Import"

### Step 5: Create Indexes (Optional)
```sql
-- Create indexes for better performance
CREATE INDEX idx_cars_make ON cars(make);
CREATE INDEX idx_cars_model ON cars(model);
CREATE INDEX idx_cars_fuel_type ON cars(fuel_type);
CREATE INDEX idx_cars_year_start ON cars(year_start);
CREATE INDEX idx_cars_display_name ON cars(display_name);
```

## 🔧 Script Features

### Error Handling
- Automatic retry on API failures
- Progress saving every 1000 cars
- Fallback logic for failed API calls

### Performance Optimizations
- Batch processing (50-100 cars per batch)
- Minimal token usage (10-20 tokens per request)
- Rate limiting to avoid quota issues
- Background processing support

### Resume Capability
All scripts support resuming from where they left off:
```bash
python generate_fuel_types.py --resume
```

## 📈 Processing Times

- **Display Names**: ~3-4 hours (34,247 cars)
- **Fuel Types**: ~45 minutes (34,247 cars)  
- **Mileage Ranges**: ~45 minutes (34,247 cars)

## 🎯 Use Cases

This enhanced database is perfect for:
- Car marketplace applications
- Vehicle recommendation systems
- Automotive data analysis
- UK car market research
- Used car valuation tools

## 📝 Notes

- All scripts use GPT-4o-mini for cost efficiency
- Progress is saved frequently to prevent data loss
- Scripts can be run in parallel on different data subsets
- Final database contains 25 columns including AI-generated content

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements to the processing scripts or documentation.