#!/usr/bin/env python3
"""
Test the sequential processor with just 3 cars to verify it's working correctly
"""

import asyncio
import pandas as pd
from pathlib import Path
from enhanced_parkers_scraper import process_url, ScrapeConfig

async def test_sequential():
    """Test sequential processing with 3 cars."""
    print("=== Testing Sequential Processor ===\n")
    
    # Configure for testing
    config = ScrapeConfig(
        headless=True,  # Run in background
        save_debug_html=False,
        timeout=30000,
        retry_count=3
    )
    
    # Read first 3 rows from the main file
    try:
        df = pd.read_excel("Series + URL.xlsx")
        test_df = df.head(3)  # Just first 3 rows
        print(f"Testing with first 3 rows from Series + URL.xlsx")
    except FileNotFoundError:
        print("❌ Error: 'Series + URL.xlsx' file not found!")
        return
    
    # Essential columns only
    essential_columns = [
        'Model URL',
        'Article_Title',
        'Pros',
        'Cons',
        'Rivals',
        'Resolved_URL',
        'Processing_Status'
    ]
    
    results_df = pd.DataFrame(columns=essential_columns)
    
    for index, row in test_df.iterrows():
        url = row['Model URL']
        print(f"--- Testing {index + 1}/3: {url} ---")
        
        # Process the URL
        result = await process_url(url, config)
        
        # Create new row with essential data only
        new_row = {
            'Model URL': url,
            'Article_Title': '',
            'Pros': '',
            'Cons': '',
            'Rivals': '',
            'Resolved_URL': '',
            'Processing_Status': result.get('processing_status', 'failed')
        }
        
        # Check if this car has a review page
        if result.get("source") == "specs_page_fallback":
            print(f"⏭️  No review page available")
            new_row['Processing_Status'] = 'skipped_no_review'
        elif result.get("processing_status") == "success":
            # Extract only the essential information
            new_row['Article_Title'] = result.get('title', '')
            new_row['Resolved_URL'] = result.get('url', '')
            
            # Get key info
            key_info = result.get('key_info', {})
            new_row['Pros'] = key_info.get('pros', '')
            new_row['Cons'] = key_info.get('cons', '')
            new_row['Rivals'] = key_info.get('rivals', '')
            
            print(f"✅ Success: {new_row['Article_Title']}")
            print(f"   Pros: {new_row['Pros'][:50]}..." if new_row['Pros'] else "   Pros: None")
            print(f"   Cons: {new_row['Cons'][:50]}..." if new_row['Cons'] else "   Cons: None")
            print(f"   Rivals: {len(new_row['Rivals'].split('|')) if new_row['Rivals'] else 0} rivals")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Add to results
        new_row_df = pd.DataFrame([new_row])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
    
    # Save test results
    test_output = "test_sequential_output.xlsx"
    results_df.to_excel(test_output, index=False)
    
    print(f"\n=== Test Complete ===")
    print(f"📁 Test output saved to: {test_output}")
    print(f"📊 Results summary:")
    print(results_df[['Article_Title', 'Processing_Status']].to_string(index=False))

if __name__ == "__main__":
    asyncio.run(test_sequential())
