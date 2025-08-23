#!/usr/bin/env python3
"""
Simple test script to process a small batch of URLs.
Use this to test the scraper before running the full production script.
"""

import asyncio
from enhanced_parkers_scraper import process_excel_file, ScrapeConfig

async def main():
    print("🧪 Testing Enhanced Parkers Scraper with small batch")
    print("=" * 50)
    
    # Test configuration
    config = ScrapeConfig(
        headless=False,  # Show browser for testing
        timeout_s=60,
        save_debug_html=True,  # Save debug files for testing
        retry_count=2
    )
    
    try:
        # Process just 5 rows for testing
        result_file = await process_excel_file(
            excel_path="Series + URL.xlsx",
            output_excel_path="test_small_batch.xlsx",
            limit=5,  # Just 5 rows
            offset=0,
            concurrency=2,  # Lower concurrency for testing
            config=config
        )
        
        print(f"\n✅ Test complete! Results saved to: {result_file}")
        print("💡 Check the output file to verify the JSON data format")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
