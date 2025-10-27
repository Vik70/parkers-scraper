# Complete Guide: Uploading to Supabase with DBeaver

## ✅ Step 1: Check Your CSV File

Your data is now prepared in `cars_for_supabase_upload.csv` with:
- ✅ Proper column names matching your schema
- ✅ Vector embeddings in text format (ready for conversion)
- ✅ All 34,247 cars with display names and fuel types
- ✅ Excluded suggested_mileage_range as requested

**File location:** `C:\Users\Vik\Documents\parkers-scraper\cars_for_supabase_upload.csv`

## ✅ Step 2: Convert Vectors in Database (IMPORTANT)

The CSV has vectors stored as text strings. You need to convert them to proper vector format in PostgreSQL.

### In DBeaver:
1. Connect to your Supabase database
2. Open a new SQL console (right-click database → SQL Editor → New SQL Script)
3. Run this to create a staging table:

```sql
-- Create staging table to import CSV data
CREATE TABLE cars_staging_temp (
    make text,
    model text,
    series_production_years text,
    year_start smallint,
    year_end text,
    real_body_type text,
    engine_type text,
    power_bhp text,
    transmission text,
    doors text,
    seats smallint,
    used_price_low_min numeric(12, 2),
    used_price_high_max numeric(12, 2),
    mpg_low_min numeric(5, 2),
    insurance_group_min smallint,
    zero_to_sixty_secs_min numeric(4, 2),
    length_mm_avg integer,
    width_mm_avg integer,
    height_mm_avg integer,
    luggage_capacity_litres_avg integer,
    essence_main text,
    image_urls text,
    display_name text,
    fuel_type text,
    specs_text text,
    vector_main_text text,
    vector_specs_text text
);
```

## ✅ Step 3: Import CSV to Staging Table

### In DBeaver:
1. Right-click on `cars_staging_temp` table → Import Data
2. Select your CSV file: `cars_for_supabase_upload.csv`
3. Check **"First line contains column names"**
4. Click **"Next"**
5. Map columns (auto-detected should work)
6. Click **"Next"** → **"Next"** → **"Start"**
7. Wait for import to complete (this may take a few minutes)

## ✅ Step 4: Convert Text Vectors to Proper Vector Type

Run this SQL to:
1. Convert text vectors to proper vector type (1536 dimensions)
2. Handle any missing values
3. Insert into your actual `cars` table

```sql
-- Convert text vectors to vector type and insert into cars table
INSERT INTO cars (
    make, model, series_production_years, year_start, year_end,
    real_body_type, engine_type, power_bhp, transmission, doors, seats,
    used_price_low_min, used_price_high_max, mpg_low_min, 
    insurance_group_min, zero_to_sixty_secs_min, length_mm_avg,
    width_mm_avg, height_mm_avg, luggage_capacity_litres_avg,
    essence_main, image_urls, vector_main, vector_specs,
    vector_main_text, vector_specs_text
)
SELECT 
    make, 
    model, 
    series_production_years, 
    year_start,
    NULLIF(year_end, '')::integer as year_end,
    real_body_type, 
    engine_type, 
    NULLIF(power_bhp, '')::integer as power_bhp, 
    transmission, 
    NULLIF(doors, '')::integer as doors, 
    seats,
    used_price_low_min, 
    used_price_high_max, 
    mpg_low_min,
    insurance_group_min, 
    zero_to_sixty_secs_min, 
    length_mm_avg,
    width_mm_avg, 
    height_mm_avg, 
    luggage_capacity_litres_avg,
    essence_main, 
    image_urls,
    -- Convert text vectors to proper vector type
    -- Remove brackets, split by comma, convert to float array, then to vector
    (string_to_array(TRIM(both '[]' from vector_main_text), ',')::float[])::vector(1536) as vector_main,
    (string_to_array(TRIM(both '[]' from vector_specs_text), ',')::float[])::vector(1536) as vector_specs,
    vector_main_text,
    vector_specs_text
FROM cars_staging_temp
WHERE vector_main_text IS NOT NULL 
  AND vector_main_text != ''
  AND vector_main_text NOT LIKE '[]'
  AND vector_specs_text IS NOT NULL
  AND vector_specs_text != ''
  AND vector_specs_text NOT LIKE '[]'
  -- Exclude rows with invalid data
  AND (year_end ~ '^[0-9]+$' OR year_end IS NULL OR year_end = '')
  AND (doors ~ '^[0-9]+$' OR doors IS NULL OR doors = '')
  AND (power_bhp ~ '^[0-9]+$' OR power_bhp IS NULL OR power_bhp = '');
```

**Alternative: Simpler INSERT (handles non-numeric values gracefully)**

If the above fails with validation errors, try this safer version:

```sql
INSERT INTO cars (
    make, model, series_production_years, year_start, year_end,
    real_body_type, engine_type, power_bhp, transmission, doors, seats,
    used_price_low_min, used_price_high_max, mpg_low_min, 
    insurance_group_min, zero_to_sixty_secs_min, length_mm_avg,
    width_mm_avg, height_mm_avg, luggage_capacity_litres_avg,
    essence_main, image_urls, vector_main, vector_specs,
    vector_main_text, vector_specs_text
)
SELECT 
    make, 
    model, 
    series_production_years, 
    year_start,
    CASE 
        WHEN year_end ~ '^[0-9]+$' THEN year_end::integer 
        ELSE NULL 
    END as year_end,
    real_body_type, 
    engine_type, 
    CASE 
        WHEN power_bhp ~ '^[0-9]+$' THEN power_bhp::integer 
        ELSE NULL 
    END as power_bhp,
    transmission, 
    CASE 
        WHEN doors ~ '^[0-9]+$' THEN doors::integer 
        ELSE NULL 
    END as doors, 
    seats,
    used_price_low_min, 
    used_price_high_max, 
    mpg_low_min,
    insurance_group_min, 
    zero_to_sixty_secs_min, 
    length_mm_avg,
    width_mm_avg, 
    height_mm_avg, 
    luggage_capacity_litres_avg,
    essence_main, 
    image_urls,
    (string_to_array(TRIM(both '[]' from vector_main_text), ',')::float[])::vector(1536) as vector_main,
    (string_to_array(TRIM(both '[]' from vector_specs_text), ',')::float[])::vector(1536) as vector_specs,
    vector_main_text,
    vector_specs_text
FROM cars_staging_temp
WHERE vector_main_text IS NOT NULL 
  AND vector_main_text != ''
  AND vector_main_text NOT LIKE '[]'
  AND vector_specs_text IS NOT NULL
  AND vector_specs_text != ''
  AND vector_specs_text NOT LIKE '[]';
```

**Important Notes:**
- This expects vectors to be in CSV format (comma-separated)
- The `text-embedding-3-small` model produces 1536-dimensional vectors
- If you get dimension errors, check vector length first:

```sql
-- Check if your vectors have correct dimensions
SELECT 
    length(vector_main_text),
    array_length(string_to_array(vector_main_text, ','), 1) as dim_check
FROM cars_staging_temp 
LIMIT 5;
```

## ✅ Step 5: Update Indexes

After insertion, rebuild the vector indexes for optimal performance:

```sql
-- Rebuild vector indexes
REINDEX INDEX cars_ivfflat_vector_main_idx;

-- Optionally, rebuild other indexes
REINDEX INDEX cars_year_range_idx;
REINDEX INDEX cars_body_type_idx;
REINDEX INDEX cars_make_model_idx;
REINDEX INDEX cars_price_range_idx;
```

## ✅ Step 6: Verify Data

Check that all data imported correctly:

```sql
-- Check row count
SELECT COUNT(*) FROM cars;

-- Check for null vectors
SELECT COUNT(*) FROM cars WHERE vector_main IS NULL;

-- Check sample data
SELECT make, model, display_name, fuel_type, 
       pg_typeof(vector_main) as vector_type,
       array_length(vector_main::float[], 1) as vector_dims
FROM cars 
LIMIT 5;
```

## ✅ Step 7: Cleanup Staging Table

Once everything is verified:

```sql
-- Drop the staging table
DROP TABLE IF EXISTS cars_staging_temp;
```

## 🔧 Troubleshooting

### Issue: "dimension mismatch" error
**Solution:** Your vectors might be a different dimension. Check with:
```sql
SELECT array_length(string_to_array(vector_main_text, ','), 1) 
FROM cars_staging_temp LIMIT 1;
```

If it's not 1536, adjust the vector dimension in the INSERT statement.

### Issue: "malformed array literal"
**Solution:** Your vector strings might have extra brackets. Update the conversion:
```sql
-- Remove brackets if they exist
string_to_array(
    TRIM(both '[]' from vector_main_text), 
    ','
)::vector(1536)
```

### Issue: Import fails with "too many columns"
**Solution:** Make sure all columns in your CSV match the staging table exactly.

### Issue: Year values don't import
**Solution:** Some year_end values might be text. The SQL above converts them properly.

## 📊 Expected Results

After successful import:
- ✅ 34,247 cars in your `cars` table
- ✅ All vectors properly indexed for fast similarity search
- ✅ Display names and fuel types populated
- ✅ All indexes ready for queries

## 🎯 Next Steps

1. **Test similarity search:**
```sql
-- Find similar cars to a car with ID 1
SELECT make, model, display_name
FROM cars
WHERE id != 1
ORDER BY vector_main <=> (SELECT vector_main FROM cars WHERE id = 1)
LIMIT 10;
```

2. **Search by fuel type:**
```sql
SELECT COUNT(*), fuel_type 
FROM cars 
GROUP BY fuel_type;
```

## 💡 Pro Tips for DBeaver

1. **Batch Import:** If CSV is too large, split it into chunks of 10,000 rows
2. **Monitor Progress:** Watch the import progress in DBeaver's bottom panel
3. **Backup First:** Always backup your existing cars table before replacing
4. **Check Logs:** If errors occur, check DBeaver's Log Viewer
5. **Vacuum:** After large imports, run `VACUUM ANALYZE cars;`

---

**Questions? Issues?** Check the error messages carefully - they usually point to the specific problem (wrong dimensions, malformed data, etc.)
