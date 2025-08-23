# Enhanced Data Organization Guide

## Overview

The enhanced Parkers scraper now organizes scraped data into **multiple, structured columns** instead of storing everything in a single JSON column. This makes the data much more accessible for analysis, filtering, and reporting.

## New Column Structure

### Basic Information
- **Title**: The car review title (e.g., "Abarth 500 Hatchback (2009-2015) review")
- **Author**: The review author (e.g., "Simon Harris")
- **Publish_Date**: Publication date (e.g., "6 June 2019")
- **Resolved_URL**: The final working review URL after resolution

### Pricing Information
- **New_Price_Min**: Minimum new car price in GBP
- **New_Price_Max**: Maximum new car price in GBP
- **Used_Price_Min**: Minimum used car price in GBP
- **Used_Price_Max**: Maximum used car price in GBP

### Running Costs
- **Road_Tax_Min**: Minimum road tax cost in GBP
- **Road_Tax_Max**: Maximum road tax cost in GBP
- **Insurance_Group**: Insurance group rating
- **Fuel_Economy**: Fuel economy information
- **Range_Min**: Minimum range in miles
- **Range_Max**: Maximum range in miles

### Vehicle Specifications
- **Doors**: Number of doors
- **Fuel_Types**: Available fuel types (e.g., "Petrol", "Diesel", "Hybrid")

### Review Content
- **Pros**: Key advantages of the vehicle
- **Cons**: Key disadvantages of the vehicle
- **Rivals**: Competitor vehicles with ratings
- **Article_Text**: Full review text content

### Technical Data
- **Full_JSON**: Complete scraped data in JSON format (for advanced users)
- **Processing_Status**: Success/failure status of the scraping process

## Data Extraction Examples

### From "At a glance" Section
```
Price new £14,380 - £15,680
Used prices £2,125 - £5,326
Road tax cost £195 - £265
Insurance group 26 - 27
Fuel economy Not tested to latest standards
Range 362 - 370 miles
Number of doors 3
Available fuel types Petrol
```

**Extracts to:**
- New_Price_Min: 14380
- New_Price_Max: 15680
- Used_Price_Min: 2125
- Used_Price_Max: 5326
- Road_Tax_Min: 195
- Road_Tax_Max: 265
- Insurance_Group: 26-27
- Fuel_Economy: Not tested to latest standards
- Range_Min: 362
- Range_Max: 370
- Doors: 3
- Fuel_Types: Petrol

### From "Pros & cons" Section
```
PROS
Distinctive looks
Appealing and fun high-performance small car
Sharp handling

CONS
More powerful Abarth Grande Punto is cheaper
Firm ride
```

**Extracts to:**
- Pros: "Distinctive looks Appealing and fun high-performance small car Sharp handling"
- Cons: "More powerful Abarth Grande Punto is cheaper Firm ride"

### From "About the author" Section
```
Written by Simon Harris Published: 6 June 2019
```

**Extracts to:**
- Author: "Simon Harris"
- Publish_Date: "6 June 2019"

### From Rivals Section
```
MINI Cooper S 4.5 out of 5
Renault Twingo Renaultsport 3.5 out of 5
Vauxhall Corsa VXR 4.0 out of 5
```

**Extracts to:**
- Rivals: "MINI Cooper S: 4.5/5 | Renault Twingo Renaultsport: 3.5/5 | Vauxhall Corsa VXR: 4.0/5"

## Benefits of New Organization

### 1. **Easy Filtering & Sorting**
- Sort by price range, insurance group, or fuel economy
- Filter by fuel type, number of doors, or author
- Find vehicles within specific price brackets

### 2. **Quick Data Access**
- No need to parse JSON for basic information
- Direct access to key metrics and specifications
- Easy to create pivot tables and charts

### 3. **Better Analysis**
- Compare vehicles across multiple dimensions
- Identify price-performance relationships
- Analyze author preferences and review patterns

### 4. **Export Flexibility**
- Export specific columns to other formats
- Create custom reports with selected data
- Integrate with other analysis tools

## Usage Examples

### Filter by Price Range
```python
import pandas as pd

# Load the enhanced Excel file
df = pd.read_excel("Series + URL_enhanced.xlsx")

# Find cars under £5,000 used
affordable_cars = df[
    (df['Used_Price_Max'] != '') & 
    (df['Used_Price_Max'].astype(float) < 5000)
]

# Find cars in specific insurance groups
low_insurance = df[df['Insurance_Group'].str.contains('1-5', na=False)]
```

### Analyze by Fuel Type
```python
# Count cars by fuel type
fuel_counts = df['Fuel_Types'].value_counts()

# Find all hybrid vehicles
hybrids = df[df['Fuel_Types'].str.contains('Hybrid', na=False, case=False)]
```

### Compare Rivals
```python
# Find cars with high rival ratings
high_rated_rivals = df[
    df['Rivals'].str.contains('5.0/5', na=False)
]
```

## Data Quality Notes

### Missing Data Handling
- Empty cells indicate data not found or not applicable
- Numeric fields are stored as strings to preserve formatting
- Use pandas' `pd.to_numeric()` for mathematical operations

### Data Validation
- All extracted data is cleaned of extra whitespace
- Common unwanted text (copyright notices, etc.) is removed
- Prices are normalized (commas removed for numeric operations)

### Error Handling
- Failed scrapes are marked with status "failed"
- Error details are preserved in the Full_JSON column
- Processing continues even if individual URLs fail

## Migration from Old Format

If you have existing data in the old single-column JSON format:

1. **Backup your current Excel file**
2. **Run the enhanced scraper** on your existing file
3. **The new columns will be added** alongside existing data
4. **Old JSON data is preserved** in the Full_JSON column
5. **New structured data** populates the organized columns

## Advanced Customization

### Adding New Columns
To extract additional data points, modify the `extract_key_info_from_sections()` function in `enhanced_parkers_scraper.py`:

```python
def extract_key_info_from_sections(sections: Dict[str, str]) -> Dict[str, str]:
    key_info = {}
    
    # Add your custom extraction logic here
    if "Your Section" in sections:
        # Extract your data
        key_info["your_field"] = extract_your_data(sections["Your Section"])
    
    return key_info
```

### Custom Data Cleaning
Modify the `clean_text()` function to handle specific text patterns:

```python
def clean_text(text: str) -> str:
    # Add your custom cleaning rules
    text = re.sub(r'Your Pattern', 'Replacement', text)
    return text.strip()
```

## Troubleshooting

### Common Issues

1. **Missing Data in Columns**
   - Check if the HTML structure has changed
   - Verify CSS selectors in the extraction functions
   - Enable debug HTML saving to inspect the source

2. **Incorrect Data Extraction**
   - Review the regex patterns in `extract_key_info_from_sections()`
   - Test with sample HTML to verify extraction logic
   - Check for variations in text formatting

3. **Performance Issues**
   - Reduce `max_rows` for testing
   - Use headless mode for production runs
   - Adjust timeout values if needed

### Debug Mode
Enable debug mode to save HTML files for inspection:

```python
config = ScrapeConfig(
    headless=False,
    save_debug_html=True,
    timeout=30000
)
```

This will create `debug_[filename].html` files that you can inspect to understand the HTML structure and adjust extraction logic accordingly.
