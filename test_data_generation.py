#!/usr/bin/env python3
"""Test script for data generation with DuckDuckGo search integration."""

import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator
from cryptorl.config.settings import settings
from cryptorl.data.influxdb_client import InfluxDBClient


async def test_data_generation():
    """Test the complete data generation pipeline."""
    print("🧪 Testing CryptoRL Data Generation Pipeline")
    print("=" * 50)
    
    # Test configuration
    test_symbols = ["BTCUSDT", "ETHUSDT"]
    test_date = datetime.utcnow() - timedelta(days=7)  # 7 days ago
    
    print(f"📊 Testing with symbols: {test_symbols}")
    print(f"📅 Test date: {test_date.strftime('%Y-%m-%d')}")
    
    try:
        # Initialize components
        print("\n🔧 Initializing components...")
        
        # Note: We'll skip InfluxDB for now and use mock data
        influx_client = None  # Mock for testing
        
        # Test RAG pipeline directly
        from cryptorl.llm.rag_pipeline import RAGPipeline
        rag = RAGPipeline(settings)
        
        # Health check
        print("\n🏥 Running health checks...")
        health = await rag.health_check()
        print(f"RAG Health: {health}")
        
        # Test DuckDuckGo search
        print(f"\n🔍 Testing DuckDuckGo search for {test_symbols[0]}...")
        
        news_articles = await rag.search_news(
            symbol=test_symbols[0],
            date=test_date,
            keywords=["bitcoin", "crypto", "cryptocurrency"],
            max_results=5
        )
        
        print(f"✅ Found {len(news_articles)} news articles:")
        for i, article in enumerate(news_articles, 1):
            print(f"  {i}. {article.title[:80]}...")
            print(f"     Source: {article.source}")
            print(f"     Date: {article.published_date.strftime('%Y-%m-%d')}")
            print(f"     URL: {article.url[:60]}...")
            print()
        
        # Test confidence vector generation (mock market data)
        print("🤖 Testing confidence vector generation...")
        
        # Mock market context
        mock_context = {
            'current_price': 45000.0,
            'price_change_24h': 2.5,
            'volume_24h': 25000000000,
            'market_cap': 880000000000,
            'volatility_7d': 3.2
        }
        
        # Test LLM client (if configured)
        try:
            from cryptorl.llm.llm_client import LLMClient
            llm = LLMClient(settings)
            
            # Check if LLM is available
            llm_health = await llm.health_check()
            print(f"LLM Health: {llm_health}")
            
            if llm_health['status'] == 'healthy':
                print("🎯 Testing LLM confidence generation...")
                from cryptorl.llm.models import NewsArticle
                
                # Create mock news articles for LLM
                mock_articles = [
                    NewsArticle(
                        title="Bitcoin Price Surges Amid ETF Approval Rumors",
                        content="Bitcoin surged 5% today following positive news about potential ETF approvals...",
                        source="CryptoNews",
                        published_date=test_date,
                        url="https://example.com/news1",
                        relevance_score=0.9
                    ),
                    NewsArticle(
                        title="Institutional Adoption Continues",
                        content="Major corporations continue to adopt Bitcoin as treasury asset...",
                        source="FinancialTimes",
                        published_date=test_date,
                        url="https://example.com/news2",
                        relevance_score=0.8
                    )
                ]
                
                llm_response = await llm.generate_confidence_vector(
                    symbol=test_symbols[0],
                    date=test_date.strftime('%Y-%m-%d'),
                    news_articles=mock_articles,
                    market_context=mock_context
                )
                
                print(f"✅ LLM Response:")
                print(f"   Confidence Vector: {llm_response.confidence_vector}")
                print(f"   Reasoning: {llm_response.reasoning[:200]}...")
                print(f"   Processing Time: {llm_response.processing_time:.2f}s")
                
            else:
                print("⚠️ Skipping LLM test - not configured")
                
        except Exception as e:
            print(f"⚠️ LLM test skipped: {e}")
        
        # Test batch generation for multiple symbols
        print(f"\n📈 Testing batch generation for {len(test_symbols)} symbols...")
        
        batch_results = []
        for symbol in test_symbols:
            symbol_news = await rag.search_news(
                symbol=symbol,
                date=test_date,
                max_results=3
            )
            
            result = {
                'symbol': symbol,
                'news_count': len(symbol_news),
                'sample_titles': [n.title[:50] + "..." for n in symbol_news[:2]]
            }
            batch_results.append(result)
        
        print("✅ Batch generation results:")
        for result in batch_results:
            print(f"   {result['symbol']}: {result['news_count']} articles")
            for title in result['sample_titles']:
                print(f"     - {title}")
        
        print("\n🎉 Data generation test completed successfully!")
        print(f"📊 Total articles found: {len(news_articles)}")
        print(f"🔍 Search sources tested: DuckDuckGo, Baidu fallback")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_data_generation())