#!/usr/bin/env python3
"""
Test script for improved pros and cons extraction
"""

import asyncio
from enhanced_parkers_scraper import process_url, ScrapeConfig

async def test_improved_extraction():
    """Test the improved pros and cons extraction."""
    
    # Test with a car that should have pros and cons
    test_url = "https://www.parkers.co.uk/vauxhall/adam/hatchback-2012/specs/"
    
    print(f"🧪 Testing improved extraction with: {test_url}")
    
    config = ScrapeConfig(
        headless=True,
        save_debug_html=False,
        timeout=30000,
        retry_count=3
    )
    
    try:
        result = await process_url(test_url, config)
        
        print(f"\n📊 Extraction Results:")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   URL: {result.get('url', 'N/A')}")
        print(f"   Source: {result.get('source', 'N/A')}")
        print(f"   Status: {result.get('processing_status', 'N/A')}")
        
        if result.get('key_info'):
            key_info = result['key_info']
            print(f"\n🔍 Key Information:")
            print(f"   Pros: {key_info.get('pros', 'None')[:100]}..." if key_info.get('pros') else "   Pros: None")
            print(f"   Cons: {key_info.get('cons', 'None')[:100]}..." if key_info.get('cons') else "   Cons: None")
            print(f"   Rivals: {key_info.get('rivals', 'None')[:100]}..." if key_info.get('rivals') else "   Rivals: None")
        
        if result.get('article_text'):
            print(f"\n📄 Article Text Length: {len(result['article_text'])} characters")
            print(f"   Preview: {result['article_text'][:200]}...")
        
        # Check if we're getting pros and cons now
        if result.get('key_info', {}).get('pros') and result.get('key_info', {}).get('cons'):
            print(f"\n✅ SUCCESS: Pros and cons extracted!")
        else:
            print(f"\n❌ Still no pros and cons found")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing Improved Pros and Cons Extraction")
    print("=" * 50)
    asyncio.run(test_improved_extraction())
