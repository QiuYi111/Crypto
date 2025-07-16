#!/usr/bin/env python3
"""Test script for SERPAPI functionality."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import httpx
from loguru import logger

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

class SerpAPITester:
    """Test SERPAPI integration for crypto news search."""
    
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_KEY not found in environment variables")
    
    async def test_search(self, query: str, date: datetime) -> Dict[str, Any]:
        """Test SERPAPI search with given query."""
        
        # Format date range (±3 days around target date)
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
            "api_key": self.api_key,
            "sort": "date",
            "date_restrict": f"d{(date_end - date_start).days}",
            "tbs": f"cdr:1,cd_min:{date_start.strftime('%m/%d/%Y')},cd_max:{date_end.strftime('%m/%d/%Y')}"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get("https://serpapi.com/search", params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract news results
                news_results = data.get("news_results", [])
                organic_results = data.get("organic_results", [])
                
                logger.info(f"SERPAPI Response Status: {response.status_code}")
                logger.info(f"News Results Found: {len(news_results)}")
                logger.info(f"Organic Results Found: {len(organic_results)}")
                
                # Combine results
                all_results = []
                
                for result in news_results + organic_results:
                    article = {
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "url": result.get("link", ""),
                        "source": result.get("source", ""),
                        "date": result.get("date", "")
                    }
                    all_results.append(article)
                
                return {
                    "success": True,
                    "query": query,
                    "date": date.strftime("%Y-%m-%d"),
                    "total_results": len(all_results),
                    "articles": all_results,
                    "raw_response": data if len(all_results) == 0 else None
                }
                
        except Exception as e:
            logger.error(f"SERPAPI search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "date": date.strftime("%Y-%m-%d")
            }
    
    async def test_crypto_searches(self):
        """Test crypto-related searches."""
        
        test_queries = [
            "Bitcoin BTC news",
            "Ethereum ETH cryptocurrency",
            "Solana SOL blockchain updates"
        ]
        
        test_date = datetime.now()
        
        logger.info("🧪 Testing SERPAPI with crypto queries...")
        
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            result = await self.test_search(query, test_date)
            
            if result["success"]:
                logger.success(f"✅ Query '{query}' returned {result['total_results']} results")
                for i, article in enumerate(result["articles"][:3]):
                    logger.info(f"  {i+1}. {article['title'][:60]}...")
            else:
                logger.error(f"❌ Query '{query}' failed: {result['error']}")
            
            print("-" * 80)

async def main():
    """Main test function."""
    
    try:
        tester = SerpAPITester()
        await tester.test_crypto_searches()
        
    except ValueError as e:
        logger.error(f"❌ Setup Error: {e}")
        logger.info("Please make sure SERPAPI_KEY is set in your .env file")
        return
    
    except Exception as e:
        logger.error(f"❌ Test Error: {e}")
        return
    
    logger.success("🎉 SERPAPI test completed!")

if __name__ == "__main__":
    asyncio.run(main())