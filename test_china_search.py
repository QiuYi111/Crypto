#!/usr/bin/env python3
"""Test script for China-compatible search APIs."""

import asyncio
import os
from datetime import datetime, timedelta
from src.cryptorl.llm.rag_pipeline import RAGPipeline
from src.cryptorl.config.settings import settings


async def test_search_apis():
    """Test the new China-compatible search APIs."""
    print("Testing China-compatible search APIs...")
    
    # Initialize RAG pipeline
    rag = RAGPipeline(settings)
    
    # Test date (7 days ago)
    test_date = datetime.utcnow() - timedelta(days=7)
    symbol = "BTCUSDT"
    
    # Check which APIs are configured
    print(f"Bing API enabled: {bool(rag.bing_api_key)}")
    print(f"Baidu API enabled: {bool(rag.baidu_api_key)}")
    print(f"SerpAPI enabled: {bool(rag.serpapi_key)}")
    print(f"Google Search enabled: {bool(rag.google_search_key and rag.google_search_cx)}")
    
    # Test search
    try:
        print(f"\nSearching for {symbol} news on {test_date.strftime('%Y-%m-%d')}...")
        
        articles = await rag.search_news(
            symbol=symbol,
            date=test_date,
            keywords=["bitcoin", "crypto", "cryptocurrency"],
            max_results=5
        )
        
        print(f"Found {len(articles)} articles:")
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article.title}")
            print(f"   Source: {article.source}")
            print(f"   Date: {article.published_date.strftime('%Y-%m-%d')}")
            print(f"   URL: {article.url}")
            print(f"   Score: {article.relevance_score}")
            print(f"   Content: {article.content[:100]}...")
            
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()
    
    # Test health check
    try:
        health = await rag.health_check()
        print(f"\nHealth check results: {health}")
    except Exception as e:
        print(f"Health check failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_search_apis())