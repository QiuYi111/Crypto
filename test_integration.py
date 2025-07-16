#!/usr/bin/env python3
"""Integration test for CryptoRL components."""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.cryptorl.llm.rag_pipeline import RAGPipeline
from src.cryptorl.llm.llm_client import LLMClient
from src.cryptorl.config.settings import settings

async def test_components():
    """Test individual components."""
    
    print("🔍 Testing CryptoRL Integration...")
    
    # Test SERPAPI
    print("\n1. Testing SERPAPI...")
    try:
        rag = RAGPipeline(settings)
        
        # Test with a simple query
        from src.cryptorl.llm.models import SearchQuery
        query = SearchQuery(
            symbol="BTCUSDT",
            date=datetime.now(),
            keywords=["news"],
            max_results=5
        )
        
        print(f"   Query: {query.to_search_string()}")
        articles = await rag.search_news("BTCUSDT", datetime.now(), ["news"], 5)
        print(f"   ✅ Found {len(articles)} articles via SERPAPI")
        
        for i, article in enumerate(articles[:2], 1):
            print(f"   {i}. {article.title[:60]}...")
            
    except Exception as e:
        print(f"   ❌ SERPAPI failed: {e}")
    
    # Test DeepSeek API
    print("\n2. Testing DeepSeek API...")
    try:
        llm_client = LLMClient(settings)
        health = await llm_client.health_check()
        print(f"   Health check: {health}")
        
        # Test with mock data
        from src.cryptorl.llm.models import NewsArticle
        mock_articles = [
            NewsArticle(
                title="Bitcoin hits new high",
                content="Bitcoin reached $65,000 amid institutional adoption",
                source="CoinDesk",
                published_date=datetime.now(),
                url="https://example.com",
                relevance_score=0.9
            )
        ]
        
        mock_context = {
            'current_price': 65000,
            'price_change_24h': 5.2,
            'volume_24h': 25000000000,
            'market_cap': 1200000000000,
            'volatility_7d': 3.1
        }
        
        response = await llm_client.generate_confidence_vector(
            symbol="BTCUSDT",
            date=datetime.now().strftime("%Y-%m-%d"),
            news_articles=mock_articles,
            market_context=mock_context
        )
        
        print(f"   ✅ DeepSeek API working: {response.confidence_vector}")
        
    except Exception as e:
        print(f"   ❌ DeepSeek API failed: {e}")

async def main():
    """Main test function."""
    load_dotenv()
    
    await test_components()
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())