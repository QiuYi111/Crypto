#!/usr/bin/env python3
"""Debug SERPAPI integration issues."""

import asyncio
import os
from datetime import datetime, timedelta
import httpx
from dotenv import load_dotenv

load_dotenv()

async def debug_serpapi():
    """Debug SERPAPI calls step by step."""
    
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("❌ SERPAPI_KEY not found")
        return
    
    # Test the exact parameters being used
    query = "Bitcoin BTC news"
    date = datetime.now()
    date_start = date - timedelta(days=3)
    date_end = date + timedelta(days=3)
    
    params = {
        "engine": "google",
        "q": query,
        "location": "United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "tbm": "nws",
        "num": 10,
        "api_key": api_key,
        "tbs": f"cdr:1,cd_min:{date_start.strftime('%m/%d/%Y')},cd_max:{date_end.strftime('%m/%d/%Y')}"
    }
    
    print("🔍 Testing SERPAPI with exact parameters...")
    print(f"Query: {query}")
    print(f"Date range: {date_start.strftime('%m/%d/%Y')} to {date_end.strftime('%m/%d/%Y')}")
    print(f"URL: https://serpapi.com/search")
    print(f"Params: {params}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("https://serpapi.com/search", params=params)
            
            print(f"\nStatus: {response.status_code}")
            print(f"Response length: {len(response.text)}")
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\nResponse keys: {list(data.keys())}")
                
                if "error" in data:
                    print(f"Error: {data['error']}")
                    return
                
                news_results = data.get("news_results", [])
                print(f"News results: {len(news_results)}")
                
                for i, result in enumerate(news_results[:2], 1):
                    print(f"\n{i}. {result.get('title', 'No title')}")
                    print(f"   Source: {result.get('source', 'Unknown')}")
                    print(f"   Date: {result.get('date', 'No date')}")
                    print(f"   URL: {result.get('link', 'No URL')}")
            else:
                print(f"Error response: {response.text[:500]}")
                
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_serpapi())