#!/usr/bin/env python3
"""
Sequential Full Database Processor for Parkers Scraper
- Processes entire Series + URL.xlsx from top to bottom
- Only extracts: Pros, Cons, Rivals, Article Title, Resolved URL
- Always appends to existing data (never overwrites)
- Remembers position and resumes from last processed row
- Processes everything in the database sequentially
"""

import asyncio
import pandas as pd
import json
from pathlib import Path
from enhanced_parkers_scraper import process_url, ScrapeConfig

# Position tracking file
POSITION_FILE = "processing_position.txt"
OUTPUT_FILE = "Series + URL_sequential_enhanced.xlsx"

def save_position(row_index: int):
    """Save current processing position."""
    with open(POSITION_FILE, 'w') as f:
        f.write(str(row_index))
    print(f"💾 Position saved: Row {row_index + 1}")

def load_position() -> int:
    """Load last processing position."""
    if Path(POSITION_FILE).exists():
        try:
            with open(POSITION_FILE, 'r') as f:
                position = int(f.read().strip())
                print(f"📍 Resuming from saved position: Row {position + 1}")
                return position
        except:
            print("⚠️  Could not read position file, starting from beginning")
            return 0
    else:
        print("📍 No previous position found, starting from beginning")
        return 0

def get_existing_urls() -> set:
    """Get URLs that have already been processed."""
    if Path(OUTPUT_FILE).exists():
        try:
            existing_df = pd.read_excel(OUTPUT_FILE)
            processed_urls = set(existing_df['Model URL'].dropna())
            print(f"📊 Found {len(processed_urls)} already processed URLs")
            return processed_urls
        except:
            print("⚠️  Could not read existing output file")
            return set()
    else:
        print("📁 No existing output file found")
        return set()

async def process_sequential_full(excel_path: str, config: ScrapeConfig, start_from_position: int = 0):
    """Process the entire Excel file sequentially from top to bottom."""
    
    # Read Excel file
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"❌ Error: '{excel_path}' file not found!")
        return
    
    total_rows = len(df)
    print(f"📊 Total rows in database: {total_rows}")
    
    # Apply quality filters to exclude old cars
    print("🔍 Applying quality filters...")
    if 'Series prod year' in df.columns:
        # Create a mask to exclude old cars
        old_cars_mask = df['Series prod year'].astype(str).str.contains(r'198\d|199\d', na=False)
        old_cars_count = old_cars_mask.sum()
        if old_cars_count > 0:
            print(f"🚫 Found {old_cars_count} old cars (1980s, 1990s) - will skip these")
            df = df[~old_cars_mask].reset_index(drop=True)
        else:
            print("✅ No old cars found to filter out")
    else:
        print("⚠️  'Series prod year' column not found, skipping old car filter")
    
    filtered_rows = len(df)
    print(f"📊 Rows after filtering: {filtered_rows}")
    
    # Get already processed URLs to avoid duplicates
    existing_urls = get_existing_urls()
    
    # Create essential columns DataFrame for new data
    essential_columns = [
        'Model URL',           # Original URL
        'Series_Production_Year',  # Series production year from original database
        # 'Article_Title',       # Title from the review page
        'Article_Text',        # Full article text from review page
        'Pros',               # Pros from review
        'Cons',               # Cons from review  
        'Rivals',             # Rivals with URLs and ratings
        'Resolved_URL',       # Final resolved URL
        # 'Processing_Status'   # Success/failed/skipped status
    ]
    
    # Check if output file exists and load it
    if Path(OUTPUT_FILE).exists():
        print(f"📁 Loading existing output file: {OUTPUT_FILE}")
        results_df = pd.read_excel(OUTPUT_FILE)
        print(f"📊 Existing database has {len(results_df)} processed cars")
    else:
        print(f"📁 Creating new output file: {OUTPUT_FILE}")
        results_df = pd.DataFrame(columns=essential_columns)
    
    # Start processing from saved position
    current_position = max(start_from_position, 0)
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"\n=== STARTING SEQUENTIAL PROCESSING ===")
    print(f"🚀 Starting from row {current_position + 1} of {filtered_rows}")
    print(f"📋 Processing {filtered_rows - current_position} remaining rows")
    print(f"🎯 Only extracting: Series Production Year, Article Title, Article Text, Pros, Cons, Rivals, Resolved URL")
    
    for index in range(current_position, filtered_rows):
        row = df.iloc[index]
        url = row['Model URL']
        
        print(f"\n--- Processing {index + 1}/{filtered_rows}: {url} ---")
        
        # Skip if already processed
        if url in existing_urls:
            print(f"⏭️  Already processed, skipping...")
            skipped += 1
            continue
        
        # Process the URL
        result = await process_url(url, config)
        
        # Create new row with essential data only
        new_row = {
            'Model URL': url,
            'Series_Production_Year': row.get('Series prod year', ''),  # Get from original database
            'Article_Title': '',
            'Article_Text': '',
            'Pros': '',
            'Cons': '',
            'Rivals': '',
            'Resolved_URL': '',
            'Processing_Status': result.get('processing_status', 'failed')
        }
        
        # Check if this car has a review page (not just specs)
        if result.get("source") == "specs_page_fallback":
            print(f"⏭️  Skipping car without review page: {url}")
            new_row['Processing_Status'] = 'skipped_no_review'
            skipped += 1
        elif result.get("processing_status") == "success":
            successful += 1
            
            # Extract only the essential information
            new_row['Article_Title'] = result.get('title', '')
            new_row['Resolved_URL'] = result.get('url', '')
            new_row['Article_Text'] = result.get('article_text', '') # Add Article_Text
            
            # Get key info
            key_info = result.get('key_info', {})
            new_row['Pros'] = key_info.get('pros', '')
            new_row['Cons'] = key_info.get('cons', '')
            new_row['Rivals'] = key_info.get('rivals', '')
            
            print(f"✅ Successfully scraped: {new_row['Article_Title']}")
            print(f"   📝 Pros: {new_row['Pros'][:50]}..." if new_row['Pros'] else "   📝 Pros: None")
            print(f"   📝 Cons: {new_row['Cons'][:50]}..." if new_row['Cons'] else "   📝 Cons: None")
            print(f"   📄 Article Text: {len(new_row['Article_Text'])} characters" if new_row['Article_Text'] else "   📄 Article Text: None")
            print(f"   🏁 Rivals: {len(new_row['Rivals'].split('|')) if new_row['Rivals'] else 0} rivals found")
        else:
            failed += 1
            print(f"❌ Failed to scrape: {result.get('error', 'Unknown error')}")
        
        # Append new row to results
        new_row_df = pd.DataFrame([new_row])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        
        # Save updated Excel file after each successful processing
        results_df.to_excel(OUTPUT_FILE, index=False)
        
        # Save position after each row
        save_position(index + 1)  # Save next position to process
        
        # Update existing URLs set
        existing_urls.add(url)
        
        # Progress update every 10 rows
        if (index + 1) % 10 == 0:
            print(f"\n📊 Progress Update:")
            print(f"   Processed: {index + 1}/{filtered_rows} ({((index + 1)/filtered_rows)*100:.1f}%)")
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            print(f"   ⏭️  Skipped: {skipped}")
    
    # Final summary
    print(f"\n=== SEQUENTIAL PROCESSING COMPLETE ===")
    print(f"📊 Total processed: {filtered_rows}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"📁 Output saved to: {OUTPUT_FILE}")
    print(f"📊 Final database contains {len(results_df)} total cars")
    
    # Clean up position file when complete
    if Path(POSITION_FILE).exists():
        Path(POSITION_FILE).unlink()
        print(f"🧹 Cleaned up position tracking file")

async def main():
    """Run the sequential full database processor."""
    print("=== Parkers Sequential Full Database Processor ===")
    print("This will process the ENTIRE database from top to bottom")
    print("🎯 Only extracting: Series Production Year, Article Text, Pros, Cons, Rivals, Resolved URL")
    print("📁 Always appends to existing data (never overwrites)")
    print("💾 Remembers position and can resume from interruptions")
    
    # Configure for production (headless mode for speed)
    config = ScrapeConfig(
        headless=True,  # Run in background for speed
        save_debug_html=False,  # Disable debug HTML to save space
        timeout=30000,  # 30 second timeout per page
        retry_count=3
    )
    
    # Check for saved position
    start_position = load_position()
    
    if start_position > 0:
        print(f"\n🔄 This will resume processing from row {start_position + 1}")
        print("   (All previous rows will be skipped)")
    
    try:
        await process_sequential_full(
            excel_path="Series + URL.xlsx",
            config=config,
            start_from_position=start_position
        )
    except FileNotFoundError:
        print("❌ Error: 'Series + URL.xlsx' file not found!")
        print("Please ensure the Excel file is in the current directory.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        # Save position even on error so we can resume
        print("💾 Position saved for resumption")

if __name__ == "__main__":
    print("⚠️  This will process the ENTIRE database sequentially.")
    print("   🎯 Only essential columns: Series Production Year, Title, Article Text, Pros, Cons, Rivals, URL")
    print("   📁 Always appends (never overwrites existing data)")
    print("   💾 Can be interrupted and resumed from last position")
    print("   ⏱️  Estimated time: 2-5 hours for full database")
    print(f"   📄 Output: {OUTPUT_FILE}")
    print(f"   💾 Position tracking: {POSITION_FILE}")
    
    response = input("\nContinue? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n🚀 Starting sequential full database processing...")
        asyncio.run(main())
    else:
        print("Operation cancelled.")
