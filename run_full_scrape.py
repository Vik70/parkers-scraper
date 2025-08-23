#!/usr/bin/env python3
"""
Production script to process the entire Series + URL.xlsx file.
This script will:
1. Read all 2736 rows from the Excel file
2. Process URLs in batches with concurrency control
3. Save progress regularly
4. Generate a comprehensive report
5. Create the final Excel file with JSON data in column C
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

from enhanced_parkers_scraper import process_excel_file, ScrapeConfig

async def main():
    print("🚗 Enhanced Parkers Scraper - Full Production Run")
    print("=" * 60)
    
    # Configuration
    input_file = "Series + URL.xlsx"
    output_file = f"Series_URL_with_JSON_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found!")
        return
    
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output file: {output_file}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration for production run
    config = ScrapeConfig(
        headless=True,  # Run in background for production
        timeout_s=60,   # Generous timeout
        save_debug_html=False,  # Don't save debug files in production
        retry_count=2
    )
    
    try:
        print("🔄 Starting full Excel processing...")
        start_time = time.time()
        
        # Process the entire file
        result_file = await process_excel_file(
            excel_path=input_file,
            output_excel_path=output_file,
            limit=None,  # Process all rows
            offset=0,    # Start from beginning
            concurrency=3,  # Conservative concurrency
            config=config
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print()
        print("✅ Processing complete!")
        print(f"📊 Output saved to: {result_file}")
        print(f"⏱️  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Generate summary report
        print("\n📈 Generating summary report...")
        df = pd.read_excel(result_file)
        
        # Count successful vs failed scrapes
        success_count = 0
        error_count = 0
        
        for idx, row in df.iterrows():
            try:
                json_data = json.loads(row.iloc[2])
                if json_data.get("processing_status") == "success":
                    success_count += 1
                else:
                    error_count += 1
            except:
                error_count += 1
        
        total_count = len(df)
        print(f"📊 Summary:")
        print(f"   Total rows: {total_count}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {error_count}")
        print(f"   Success rate: {(success_count/total_count)*100:.1f}%")
        
        # Save summary report
        summary_file = f"scrape_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Enhanced Parkers Scraper - Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Output file: {result_file}\n")
            f.write(f"Total time: {duration:.1f} seconds\n")
            f.write(f"Total rows: {total_count}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {error_count}\n")
            f.write(f"Success rate: {(success_count/total_count)*100:.1f}%\n")
        
        print(f"📄 Summary report saved to: {summary_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting production scraper...")
    print("💡 Tip: This will take several hours for 2736 URLs.")
    print("💡 You can stop anytime with Ctrl+C and resume later.")
    print()
    
    # Ask for confirmation
    response = input("Continue with full scrape? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        asyncio.run(main())
    else:
        print("❌ Cancelled by user")
