#!/usr/bin/env python3
"""Synchronous test for SERPAPI with proper query strings."""

import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def test_serpapi_integration():
    """Test SERPAPI with actual CryptoRL query format."""
    
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("❌ SERPAPI_KEY not found")
        return False
    
    # Test actual CryptoRL query format
    symbol = "BTCUSDT"
    date = datetime.now()
    
    # This is what CryptoRL generates
    base_query = f"{symbol.replace('USDT', '')} cryptocurrency"
    
    # Format dates for SERPAPI
    date_start = date - timedelta(days=3)
    date_end = date + timedelta(days=3)
    
    params = {
        "engine": "google",
        "q": base_query,
        "location": "United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "tbm": "nws",
        "num": 10,
        "api_key": api_key,
        "tbs": f"cdr:1,cd_min:{date_start.strftime('%m/%d/%Y')},cd_max:{date_end.strftime('%m/%d/%Y')}"
    }
    
    print("🔍 Testing CryptoRL SERPAPI integration...")
    print(f"Query: {base_query}")
    print(f"Date range: {date_start.strftime('%m/%d/%Y')} to {date_end.strftime('%m/%d/%Y')}")
    
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        
        print(f"\nStatus: {response.status_code}")
        print(f"Response size: {len(response.text)} bytes")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for errors in response
            if "error" in data:
                print(f"❌ API Error: {data['error']}")
                return False
            
            news_results = data.get("news_results", [])
            print(f"✅ Found {len(news_results)} news articles")
            
            if len(news_results) > 0:
                print("\nTop 3 articles:")
                for i, result in enumerate(news_results[:3], 1):
                    print(f"{i}. {result.get('title', 'No title')}")
                    print(f"   Source: {result.get('source', 'Unknown')}")
                    print(f"   Date: {result.get('date', 'No date')}")
                return True
            else:
                print("❌ No news results found")
                
                # Check what keys we got
                print(f"Available keys: {list(data.keys())}")
                if "organic_results" in data:
                    print(f"Organic results: {len(data['organic_results'])}")
                
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    success = test_serpapi_integration()
    if success:
        print("\n🎉 SERPAPI integration test PASSED")
    else:
        print("\n❌ SERPAPI integration test FAILED")