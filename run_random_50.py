#!/usr/bin/env python3
"""
Script to run the enhanced Parkers scraper with 50 random rows from the database.
This provides a good sample size for testing while keeping processing time reasonable.
"""

import asyncio
import json
import pandas as pd
import random
from pathlib import Path
from enhanced_parkers_scraper import process_excel_file, ScrapeConfig

async def process_random_rows(excel_path: str, config: ScrapeConfig, num_random_rows: int = 10, append_mode: bool = False):
    """Process a specified number of random rows from the Excel file."""
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Ensure we have the required columns
    if 'Model URL' not in df.columns:
        print("Error: 'Model URL' column not found in Excel file")
        return
    
    total_rows = len(df)
    print(f"Total rows in database: {total_rows}")
    
    # Apply quality filters to exclude old cars and cars without review pages
    print("🔍 Applying quality filters...")
    
    # Filter 1: Exclude old cars (1980s, 1990s)
    if 'Series prod year' in df.columns:
        # Create a mask to exclude old cars
        old_cars_mask = df['Series prod year'].astype(str).str.contains(r'198\d|199\d', na=False)
        old_cars_count = old_cars_mask.sum()
        if old_cars_count > 0:
            print(f"🚫 Filtering out {old_cars_count} old cars (1980s, 1990s)")
            df = df[~old_cars_mask]
        else:
            print("✅ No old cars found to filter out")
    else:
        print("⚠️  'Series prod year' column not found, skipping old car filter")
    
    # Filter 2: Check which cars have review pages (we'll do this during processing)
    print("🔍 Will check for review page availability during processing")
    
    if append_mode:
        # Check if we have an existing enhanced database to append to
        existing_enhanced_path = excel_path.replace('.xlsx', '_random_10_enhanced.xlsx')
        if Path(existing_enhanced_path).exists():
            print(f"📁 Found existing enhanced database: {existing_enhanced_path}")
            existing_df = pd.read_excel(existing_enhanced_path)
            print(f"📊 Existing database has {len(existing_df)} cars with data")
            
            # Get URLs that are already processed
            processed_urls = set(existing_df['Model URL'].dropna())
            print(f"🔍 Found {len(processed_urls)} already processed URLs")
            
            # Filter out already processed URLs
            unprocessed_df = df[~df['Model URL'].isin(processed_urls)]
            print(f"🆕 {len(unprocessed_df)} URLs remaining to process")
            
            if len(unprocessed_df) < num_random_rows:
                print(f"⚠️  Only {len(unprocessed_df)} unprocessed URLs remaining, will process all of them.")
                num_random_rows = len(unprocessed_df)
            
            # Select random rows from unprocessed URLs
            random_indices = random.sample(range(len(unprocessed_df)), num_random_rows)
            random_df = unprocessed_df.iloc[random_indices].copy().reset_index(drop=True)
            
            print(f"✅ Selected {num_random_rows} random unprocessed rows for processing")
            print(f"Random row indices from unprocessed: {sorted(random_indices)}")
        else:
            print("📁 No existing enhanced database found, starting fresh")
            append_mode = False
    
    if not append_mode:
        # Original logic for fresh start
        if total_rows < num_random_rows:
            print(f"⚠️  Warning: Database only has {total_rows} rows, will process all of them.")
            num_random_rows = total_rows
        
        # Select random rows
        random_indices = random.sample(range(len(df)), num_random_rows)
        random_df = df.iloc[random_indices].copy().reset_index(drop=True)
        
        print(f"✅ Selected {num_random_rows} random rows for processing")
        print(f"Random row indices: {sorted(random_indices)}")
    
    # Add new columns for structured data
    new_columns = [
        'Title', 'Author', 'Publish_Date', 'New_Price_Min', 'New_Price_Max',
        'Used_Price_Min', 'Used_Price_Max', 'Road_Tax_Min', 'Road_Tax_Max',
        'Insurance_Group', 'Fuel_Economy', 'Range_Min', 'Range_Max', 'Doors',
        'Fuel_Types', 'Pros', 'Cons', 'Rivals', 'Article_Text', 'Full_JSON',
        'Processing_Status', 'Resolved_URL'
    ]
    
    for col in new_columns:
        if col not in random_df.columns:
            random_df[col] = ''
    
    # Process URLs
    total_urls = len(random_df)
    successful = 0
    failed = 0
    
    for index, row in random_df.iterrows():
        url = row['Model URL']
        if pd.isna(url) or not url:
            print(f"Row {index + 1}: No URL found")
            continue
        
        print(f"\nProcessing random row {index + 1}/{total_urls}: {url}")
        
        # Import the process_url function here to avoid circular imports
        from enhanced_parkers_scraper import process_url
        
        # Process the URL
        result = await process_url(url, config)
        
        # Check if this car has a review page (not just specs)
        if result.get("source") == "specs_page_fallback":
            print(f"⏭️  Skipping car without review page: {url}")
            random_df.at[index, 'Processing_Status'] = 'skipped_no_review'
            random_df.at[index, 'Full_JSON'] = json.dumps(result, indent=2)
            continue  # Skip to next car
        
        # Update DataFrame with results
        if result.get("processing_status") == "success":
            successful += 1
            
            # Extract key information
            key_info = result.get("key_info", {})
            
            random_df.at[index, 'Title'] = result.get('title', '')
            random_df.at[index, 'Author'] = key_info.get('author', '')
            random_df.at[index, 'Publish_Date'] = key_info.get('publish_date', '')
            random_df.at[index, 'New_Price_Min'] = key_info.get('new_price_min', '')
            random_df.at[index, 'New_Price_Max'] = key_info.get('new_price_max', '')
            random_df.at[index, 'Used_Price_Min'] = key_info.get('used_price_min', '')
            random_df.at[index, 'Used_Price_Max'] = key_info.get('used_price_max', '')
            random_df.at[index, 'Road_Tax_Min'] = key_info.get('road_tax_min', '')
            random_df.at[index, 'Road_Tax_Max'] = key_info.get('road_tax_max', '')
            random_df.at[index, 'Insurance_Group'] = key_info.get('insurance_group', '')
            random_df.at[index, 'Fuel_Economy'] = key_info.get('fuel_economy', '')
            random_df.at[index, 'Range_Min'] = key_info.get('range_min', '')
            random_df.at[index, 'Range_Max'] = key_info.get('range_max', '')
            random_df.at[index, 'Doors'] = key_info.get('doors', '')
            random_df.at[index, 'Fuel_Types'] = key_info.get('fuel_types', '')
            random_df.at[index, 'Pros'] = key_info.get('pros', '')
            random_df.at[index, 'Cons'] = key_info.get('cons', '')
            random_df.at[index, 'Rivals'] = key_info.get('rivals', '')
            random_df.at[index, 'Article_Text'] = result.get('article_text', '')
            random_df.at[index, 'Full_JSON'] = json.dumps(result, indent=2)
            random_df.at[index, 'Processing_Status'] = 'success'
            random_df.at[index, 'Resolved_URL'] = result.get('url', '')
            
            print(f"✓ Successfully scraped: {result.get('title', 'N/A')}")
            
        else:
            failed += 1
            random_df.at[index, 'Processing_Status'] = 'failed'
            random_df.at[index, 'Full_JSON'] = json.dumps(result, indent=2)
            print(f"✗ Failed to scrape: {result.get('error', 'Unknown error')}")
    
    # Save updated Excel file
    output_path = excel_path.replace('.xlsx', '_random_10_enhanced.xlsx')
    
    if append_mode and Path(output_path).exists():
        # Append mode: combine existing data with new data
        print(f"📁 Appending {len(random_df)} new cars to existing database...")
        existing_df = pd.read_excel(output_path)
        
        # Combine existing and new data
        combined_df = pd.concat([existing_df, random_df], ignore_index=True)
        combined_df.to_excel(output_path, index=False)
        
        print(f"📊 Database now contains {len(combined_df)} total cars")
        print(f"🆕 Added {len(random_df)} new cars to existing {len(existing_df)} cars")
    else:
        # Fresh start: save new data
        random_df.to_excel(output_path, index=False)
        print(f"📁 Created new enhanced database with {len(random_df)} cars")
    
    # Count skipped cars (no review page)
    skipped_count = len(random_df[random_df['Processing_Status'] == 'skipped_no_review'])
    
    print(f"\n=== RANDOM SAMPLE COMPLETE ===")
    print(f"Total URLs processed: {total_urls}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped (no review page): {skipped_count}")
    print(f"📁 Output saved to: {output_path}")
    
    if skipped_count > 0:
        print(f"\n💡 Quality Note: {skipped_count} cars were skipped because they don't have review pages")
        print("   This ensures you only get real review content, not just specs data")
    
    # Also save the random indices for reference
    indices_file = "random_10_indices.txt"
    with open(indices_file, 'w') as f:
        if append_mode:
            f.write(f"Appended {num_random_rows} new cars to existing database\n")
            f.write(f"New cars selected from unprocessed URLs\n")
        else:
            f.write(f"Random {num_random_rows} rows selected from total {total_rows} rows\n")
        f.write(f"Selected row indices: {sorted(random_indices)}\n")
        f.write(f"Original row numbers: {[i+1 for i in sorted(random_indices)]}\n")
    
    print(f"Random row indices saved to: {indices_file}")

async def main(append_mode: bool = False):
    """Run the enhanced scraper on 10 random rows."""
    if append_mode:
        print("=== Enhanced Parkers Scraper - Append 10 New Cars ===")
        print("This will process 10 new random rows and append them to your existing database.")
    else:
        print("=== Enhanced Parkers Scraper - Random 10 Rows ===")
        print("This will process 10 randomly selected rows from your database.")
    
    # Configure for production (headless mode for speed)
    config = ScrapeConfig(
        headless=True,  # Run in background for speed
        save_debug_html=False,  # Disable debug HTML to save space
        timeout=30000,  # 30 second timeout per page
        retry_count=3
    )
    
    try:
        # Process the random sample
        await process_random_rows(
            excel_path="Series + URL.xlsx",
            config=config,
            num_random_rows=10,
            append_mode=append_mode
        )
        
    except FileNotFoundError:
        print("❌ Error: 'Series + URL.xlsx' file not found!")
        print("Please ensure the Excel file is in the current directory.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Add user confirmation for production run
    print("⚠️  This will process 10 random rows from your database.")
    print("   Estimated time: 3-8 minutes depending on website response")
    print("   Output: Series + URL_random_10_enhanced.xlsx")
    print("   Random indices will be saved to: random_10_indices.txt")
    
    # Check if append mode is available
    existing_file = Path("Series + URL_random_10_enhanced.xlsx")
    if existing_file.exists():
        print(f"\n📁 Found existing database with data: {existing_file}")
        print("   You can choose to:")
        print("   1. Append new cars (recommended)")
        print("   2. Start fresh (overwrites existing data)")
        
        mode_choice = input("\nChoose mode (1=append, 2=fresh): ").strip()
        append_mode = mode_choice == "1"
        
        if append_mode:
            print("✅ Append mode selected - new cars will be added to existing database")
        else:
            print("🔄 Fresh start mode selected - existing database will be overwritten")
    else:
        print("\n📁 No existing database found - will create new one")
        append_mode = False
    
    response = input("\nContinue? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print("\nStarting random sample processing...")
        asyncio.run(main(append_mode=append_mode))
    else:
        print("Operation cancelled.")
