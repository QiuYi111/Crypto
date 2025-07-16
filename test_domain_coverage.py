#!/usr/bin/env python3
"""Test script to verify domain-specific search coverage."""

import asyncio
from datetime import datetime, timedelta
from src.cryptorl.llm.langsearch_client import LangSearchClient
from src.cryptorl.config.settings import Settings

async def test_domain_coverage():
    """Test that all 7 confidence vector domains are covered."""
    
    settings = Settings()
    client = LangSearchClient(settings)
    
    print("🔍 Testing Domain-Specific Search Coverage")
    print("=" * 50)
    
    # Test symbol and date
    symbol = "BTCUSDT"
    test_date = datetime.utcnow() - timedelta(days=1)
    
    # Test all domains
    domains = [
        "fundamentals", "industry_condition", "geopolitics",
        "macroeconomics", "technical_sentiment", "regulatory_impact", "innovation_impact"
    ]
    
    print(f"Testing coverage for {symbol} on {test_date.strftime('%Y-%m-%d')}")
    print()
    
    # Search all domains
    results = await client.search_all_domains(
        symbol=symbol,
        date=test_date,
        max_results_per_domain=2
    )
    
    # Check coverage
    coverage_summary = {}
    total_articles = 0
    
    for domain, articles in results.items():
        count = len(articles)
        total_articles += count
        coverage_summary[domain] = count
        
        print(f"📊 {domain.upper().replace('_', ' ')}: {count} articles")
        for i, article in enumerate(articles[:2], 1):
            print(f"   {i}. {article.title[:60]}...")
        print()
    
    # Test global context
    print("🌍 Testing Global Context Search")
    global_articles = await client.search_global_context(
        date=test_date,
        max_results=5
    )
    
    print(f"Global context articles: {len(global_articles)}")
    for i, article in enumerate(global_articles[:3], 1):
        print(f"   {i}. {article.title[:60]}...")
    
    print()
    print("📈 Coverage Summary:")
    print("=" * 30)
    
    all_covered = all(count > 0 for count in coverage_summary.values())
    
    for domain, count in coverage_summary.items():
        status = "✅" if count > 0 else "❌"
        print(f"{status} {domain}: {count} articles")
    
    print(f"\nTotal articles across all domains: {total_articles}")
    print(f"Global context articles: {len(global_articles)}")
    print(f"All domains covered: {'✅ YES' if all_covered else '❌ NO'}")
    
    return all_covered

if __name__ == "__main__":
    asyncio.run(test_domain_coverage())