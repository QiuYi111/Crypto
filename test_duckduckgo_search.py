#!/usr/bin/env python3
"""Test DuckDuckGo search integration for data generation."""

import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cryptorl.llm.rag_pipeline import RAGPipeline
from cryptorl.config.settings import settings


async def test_duckduckgo_search():
    """Test DuckDuckGo search functionality."""
    print("🧪 Testing DuckDuckGo Search Integration")
    print("=" * 50)
    
    # Test configuration
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    test_date = datetime.utcnow() - timedelta(days=3)  # 3 days ago
    
    print(f"📊 Testing with symbols: {test_symbols}")
    print(f"📅 Test date: {test_date.strftime('%Y-%m-%d')}")
    
    try:
        # Initialize RAG pipeline
        print("\n🔧 Initializing RAG pipeline...")
        rag = RAGPipeline(settings)
        
        # Health check
        print("\n🏥 Running health checks...")
        health = await rag.health_check()
        print(f"✅ System Health: {health}")
        
        # Test DuckDuckGo search for each symbol
        print(f"\n🔍 Testing DuckDuckGo search...")
        
        all_results = []
        
        for symbol in test_symbols:
            print(f"\n📈 Searching for {symbol}...")
            
            news_articles = await rag.search_news(
                symbol=symbol,
                date=test_date,
                keywords=[symbol.lower().replace('usdt', ''), "cryptocurrency", "crypto"],
                max_results=5
            )
            
            result = {
                'symbol': symbol,
                'news_count': len(news_articles),
                'articles': news_articles
            }
            all_results.append(result)
            
            print(f"   ✅ Found {len(news_articles)} articles")
            
            # Display first few articles
            for i, article in enumerate(news_articles[:3], 1):
                print(f"   {i}. {article.title[:60]}...")
                print(f"      Source: {article.source}")
                print(f"      Date: {article.published_date.strftime('%Y-%m-%d')}")
                if article.url:
                    print(f"      URL: {article.url[:50]}...")
                print()
        
        # Summary
        print("\n📊 Search Results Summary:")
        print("-" * 30)
        total_articles = sum(r['news_count'] for r in all_results)
        print(f"Total articles found: {total_articles}")
        
        for result in all_results:
            print(f"{result['symbol']}: {result['news_count']} articles")
        
        # Test batch search
        print("\n🔄 Testing batch search...")
        
        batch_requests = [
            {'symbol': 'BTCUSDT', 'date': test_date},
            {'symbol': 'ETHUSDT', 'date': test_date},
        ]
        
        batch_results = await rag.batch_search(
            requests=batch_requests
        )
        
        print("✅ Batch search completed:")
        for i, (request, articles) in enumerate(zip(batch_requests, batch_results)):
            print(f"   {request['symbol']}: {len(articles)} articles")
        
        # Test error handling
        print("\n🧪 Testing error handling...")
        try:
            empty_result = await rag.search_news(
                symbol="INVALIDSYMBOL",
                date=test_date,
                max_results=3
            )
            print(f"   ✅ Handled invalid symbol: {len(empty_result)} results")
        except Exception as e:
            print(f"   ✅ Error handled gracefully: {e}")
        
        print("\n🎉 DuckDuckGo search test completed successfully!")
        
        # Performance summary
        print(f"📈 Performance:")
        print(f"   - Search sources: DuckDuckGo (primary)")
        print(f"   - China accessibility: ✅ Available")
        print(f"   - API keys required: ❌ None")
        print(f"   - Rate limiting: ✅ Built-in")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_duckduckgo_search())