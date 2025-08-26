#!/usr/bin/env python3
"""
Test script to try the improved scraper with just 2-3 cars
"""

import asyncio
import pandas as pd
from enhanced_parkers_scraper import process_url, ScrapeConfig

async def test_couple_cars():
    """Test the improved scraper with 2-3 cars."""
    print("=== Testing Improved Parkers Scraper ===")
    print("Now scraping specs pages directly for better data quality!\n")
    
    # Configure for testing (non-headless to see what's happening)
    config = ScrapeConfig(
        headless=False,  # Show browser for debugging
        save_debug_html=True,  # Save HTML for inspection
        timeout=30000,  # 30 second timeout
        retry_count=3
    )
    
    # Test with 3 different cars
    test_urls = [
        "https://www.parkers.co.uk/abarth/500/hatchback-2009/specs/",
        "https://www.parkers.co.uk/volvo/v60/estate-2018/specs/",
        "https://www.parkers.co.uk/subaru/forester/estate-2020/specs/"
    ]
    
    results = []
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n{'='*60}")
        print(f"TESTING CAR {i}/3: {url}")
        print(f"{'='*60}")
        
        try:
            # Process the URL
            result = await process_url(url, config)
            
            if result.get("processing_status") == "success":
                print(f"✅ SUCCESS!")
                print(f"Title: {result.get('title', 'N/A')}")
                
                # Show key info
                key_info = result.get("key_info", {})
                if key_info:
                    print(f"Author: {key_info.get('author', 'N/A')}")
                    print(f"New Price: £{key_info.get('new_price_min', 'N/A')} - £{key_info.get('new_price_max', 'N/A')}")
                    print(f"Used Price: £{key_info.get('used_price_min', 'N/A')} - £{key_info.get('used_price_max', 'N/A')}")
                    print(f"Pros: {key_info.get('pros', 'N/A')[:100]}...")
                    print(f"Cons: {key_info.get('cons', 'N/A')[:100]}...")
                
                # Show article text preview
                article_text = result.get("article_text", "")
                if article_text:
                    print(f"Article Preview: {article_text[:200]}...")
                
                results.append({
                    "url": url,
                    "status": "success",
                    "title": result.get("title", ""),
                    "data_quality": "good" if len(article_text) > 100 else "poor"
                })
                
            else:
                print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
                results.append({
                    "url": url,
                    "status": "failed",
                    "error": result.get("error", "Unknown error")
                })
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            results.append({
                "url": url,
                "status": "exception",
                "error": str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    print(f"Total tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\nData Quality Assessment:")
        for result in results:
            if result["status"] == "success":
                print(f"✅ {result['title']}: {result['data_quality']} quality")
    
    print(f"\nThe improved scraper now:")
    print("✅ Scrapes specs pages directly (no more navigation failures)")
    print("✅ Extracts review content embedded in specs pages")
    print("✅ Provides much better data quality")
    print("✅ Avoids timeout issues from trying to load separate review pages")

if __name__ == "__main__":
    asyncio.run(test_couple_cars())
