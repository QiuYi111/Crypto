"""Test script to verify LangSearch integration."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptorl.config.settings import Settings
from cryptorl.llm.langsearch_client import LangSearchClient
from cryptorl.llm.rag_pipeline import RAGPipeline


async def test_langsearch_client():
    """Test the LangSearch client directly."""
    print("🧪 Testing LangSearch Client...")
    
    settings = Settings()
    
    if not settings.langsearch_api_key:
        print("❌ LangSearch API key not configured")
        print("   Please add LANGSEARCH_API_KEY to your .env file")
        return False
    
    client = LangSearchClient(settings)
    
    # Test health check
    health = await client.health_check()
    print(f"   Health check: {health}")
    
    if health["status"] != "healthy":
        print("❌ LangSearch client health check failed")
        return False
    
    # Test search
    test_date = datetime.utcnow() - timedelta(days=1)
    print(f"   Testing search for BTCUSDT on {test_date.strftime('%Y-%m-%d')}...")
    
    try:
        articles = await client.search_news(
            symbol="BTCUSDT",
            date=test_date,
            max_results=3,
            freshness="oneWeek"
        )
        
        print(f"   ✅ Found {len(articles)} articles")
        for i, article in enumerate(articles[:2], 1):
            print(f"   {i}. {article.title[:100]}...")
            print(f"      Source: {article.source}")
            print(f"      Relevance: {article.relevance_score:.2f}")
        
        if articles:
            # Test reranking
            print("   Testing semantic reranking...")
            reranked_articles = await client.rerank_articles(
                query="Bitcoin price impact crypto trading",
                articles=articles,
                top_n=3
            )
            
            print(f"   ✅ Reranked {len(reranked_articles)} articles")
            for i, article in enumerate(reranked_articles[:2], 1):
                print(f"   {i}. Score: {article.relevance_score:.3f} - {article.title[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


async def test_rag_pipeline():
    """Test the RAG pipeline with LangSearch integration."""
    print("🧪 Testing RAG Pipeline...")
    
    settings = Settings()
    pipeline = RAGPipeline(settings)
    
    # Test health check
    health = await pipeline.health_check()
    print(f"   RAG Health: {health}")
    
    test_date = datetime.utcnow() - timedelta(days=1)
    print(f"   Testing RAG search for ETHUSDT on {test_date.strftime('%Y-%m-%d')}...")
    
    try:
        articles = await pipeline.search_news(
            symbol="ETHUSDT",
            date=test_date,
            max_results=3
        )
        
        print(f"   ✅ Found {len(articles)} articles via RAG pipeline")
        for i, article in enumerate(articles[:2], 1):
            print(f"   {i}. {article.title[:100]}...")
            print(f"      Source: {article.source}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        return False


async def test_confidence_generation():
    """Test confidence generation with LangSearch news."""
    print("🧪 Testing Confidence Generation...")
    
    from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator
    
    settings = Settings()
    generator = ConfidenceVectorGenerator(settings)
    
    test_date = datetime.utcnow() - timedelta(days=1)
    print(f"   Testing confidence generation for SOLUSDT on {test_date.strftime('%Y-%m-%d')}...")
    
    try:
        confidence = await generator.generate_daily_confidence(
            symbol="SOLUSDT",
            date=test_date,
            use_cached=False
        )
        
        if confidence:
            print(f"   ✅ Generated confidence vector:")
            print(f"      Fundamentals: {confidence.fundamentals:.3f}")
            print(f"      Industry: {confidence.industry_condition:.3f}")
            print(f"      Geopolitics: {confidence.geopolitics:.3f}")
            print(f"      Macroeconomics: {confidence.macroeconomics:.3f}")
            print(f"      Technical: {confidence.technical_sentiment:.3f}")
            print(f"      Regulatory: {confidence.regulatory_impact:.3f}")
            print(f"      Innovation: {confidence.innovation_impact:.3f}")
            print(f"      News sources: {len(confidence.news_sources)} sources")
            return True
        else:
            print("⚠️  No confidence vector generated (no news found)")
            return True
            
    except Exception as e:
        print(f"❌ Confidence generation test failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    print("🚀 LangSearch Integration Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: LangSearch client
    results.append(await test_langsearch_client())
    print()
    
    # Test 2: RAG pipeline
    results.append(await test_rag_pipeline())
    print()
    
    # Test 3: Confidence generation
    results.append(await test_confidence_generation())
    print()
    
    # Summary
    print("📊 Test Summary:")
    print("=" * 30)
    tests = ["LangSearch Client", "RAG Pipeline", "Confidence Generation"]
    
    for test_name, result in zip(tests, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! LangSearch integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())