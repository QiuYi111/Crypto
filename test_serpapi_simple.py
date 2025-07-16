#!/usr/bin/env python3
"""Simple test script for SERPAPI functionality."""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_serpapi():
    """Quick SERPAPI test."""
    api_key = os.getenv("SERPAPI_KEY")
    
    if not api_key:
        print("❌ SERPAPI_KEY not found in .env file")
        return
    
    print("🔍 Testing SERPAPI connection...")
    
    params = {
        "engine": "google",
        "q": "Bitcoin BTC news today",
        "tbm": "nws",
        "num": 5,
        "api_key": api_key
    }
    
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            news_results = data.get("news_results", [])
            
            print(f"✅ Found {len(news_results)} news articles")
            
            for i, result in enumerate(news_results[:3], 1):
                print(f"{i}. {result.get('title', 'No title')}")
                print(f"   Source: {result.get('source', 'Unknown')}")
                print(f"   URL: {result.get('link', 'No URL')}")
                print()
                
        elif response.status_code == 401:
            print("❌ Invalid SERPAPI_KEY")
        else:
            print(f"❌ Error: {response.status_code} - {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_serpapi()