#!/usr/bin/env python3
"""Simple test for DuckDuckGo search functionality."""

import asyncio
from datetime import datetime, timedelta
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from file to avoid module issues
from src.cryptorl.llm.rag_pipeline import RAGPipeline
from src.cryptorl.config.settings import Settings


async def test_duckduckgo_simple():
    """Test DuckDuckGo search with direct imports."""
    print("🧪 Testing DuckDuckGo Search (Simple Test)")
    print("=" * 50)
    
    try:
        # Initialize settings
        settings = Settings()
        
        # Initialize RAG pipeline
        print("🔧 Initializing RAG pipeline...")
        rag = RAGPipeline(settings)
        
        # Test date
        test_date = datetime.utcnow() - timedelta(days=2)
        symbol = "BTCUSDT"
        
        print(f"📅 Testing {symbol} on {test_date.strftime('%Y-%m-%d')}")
        
        # Test DuckDuckGo search
        print("🔍 Searching DuckDuckGo for crypto news...")
        
        news_articles = await rag.search_news(
            symbol=symbol,
            date=test_date,
            keywords=["bitcoin", "crypto", "cryptocurrency"],
            max_results=3
        )
        
        print(f"✅ Found {len(news_articles)} articles:")
        
        for i, article in enumerate(news_articles, 1):
            print(f"\n{i}. {article.title}")
            print(f"   Source: {article.source}")
            print(f"   Date: {article.published_date}")
            print(f"   URL: {article.url}")
            print(f"   Content: {article.content[:100]}...")
        
        # Test health check
        print("\n🏥 Health check:")
        health = await rag.health_check()
        print(json.dumps(health, indent=2))
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_duckduckgo_simple())