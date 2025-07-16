#!/usr/bin/env python3
"""Direct test of DuckDuckGo search functionality."""

import asyncio
import httpx
from datetime import datetime, timedelta


async def test_duckduckgo_direct():
    """Test DuckDuckGo search directly using HTTP."""
    print("🧪 Testing DuckDuckGo Search Directly")
    print("=" * 50)
    
    try:
        # Test DuckDuckGo instant answer API
        search_url = "https://api.duckduckgo.com/"
        
        params = {
            "q": "bitcoin cryptocurrency news",
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
            "t": "cryptorl-test"
        }
        
        print("🔍 Testing DuckDuckGo API...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            print(f"✅ API Response Status: {response.status_code}")
            print(f"🔍 Found {len(data.get('RelatedTopics', []))} related topics")
            
            # Show some results
            topics = data.get('RelatedTopics', [])
            for i, topic in enumerate(topics[:3], 1):
                if isinstance(topic, dict) and 'Text' in topic:
                    print(f"{i}. {topic.get('Text', 'No text')[:100]}...")
                    if 'FirstURL' in topic:
                        print(f"   URL: {topic.get('FirstURL')}")
                    print()
        
        # Test web scraping approach
        print("🌐 Testing web scraping approach...")
        
        search_url = "https://html.duckduckgo.com/html/"
        params = {
            "q": "bitcoin news latest",
            "t": "h_"
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(search_url, params=params, headers=headers)
            
            print(f"✅ Web Response Status: {response.status_code}")
            print(f"📄 Response length: {len(response.text)} characters")
            
            # Basic content check
            if "bitcoin" in response.text.lower():
                print("✅ Successfully retrieved search results")
                
                # Count results
                result_count = response.text.count('result__a')
                print(f"🔍 Found approximately {result_count} search results")
            else:
                print("⚠️ No obvious search results found")
        
        print("\n🎉 DuckDuckGo search test completed!")
        print("✅ Both API and web scraping approaches tested")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Try a simpler approach
        print("\n🔧 Trying simpler approach...")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("https://duckduckgo.com/?q=bitcoin+news&format=json")
                print(f"Simple test status: {response.status_code}")
        except Exception as e2:
            print(f"Simple test also failed: {e2}")


if __name__ == "__main__":
    asyncio.run(test_duckduckgo_direct())