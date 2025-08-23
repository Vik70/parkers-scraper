#!/usr/bin/env python3
"""
Test script for the enhanced Parkers scraper with organized data columns.
"""

import asyncio
from enhanced_parkers_scraper import process_excel_file, ScrapeConfig

async def main():
    """Test the enhanced scraper with a small batch."""
    print("=== Testing Enhanced Parkers Scraper ===")
    print("This will process a few rows to demonstrate the new column organization.")
    
    # Configure for testing (non-headless to see what's happening)
    config = ScrapeConfig(
        headless=False,  # Show browser for debugging
        save_debug_html=True,  # Save HTML for inspection
        timeout=30000
    )
    
    # Process just 2 rows for testing
    await process_excel_file(
        excel_path="Series + URL.xlsx",
        config=config,
        max_rows=2  # Limit to 2 rows for quick testing
    )
    
    print("\n=== Test Complete ===")
    print("Check the generated '_enhanced.xlsx' file to see the organized columns!")

if __name__ == "__main__":
    asyncio.run(main())
