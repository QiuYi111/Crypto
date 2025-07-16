#!/usr/bin/env python3
"""Test script for DeepSeek integration."""

import asyncio
import os
from datetime import datetime
from loguru import logger

from src.cryptorl.config.settings import settings
from src.cryptorl.llm.llm_factory import LLMFactory
from src.cryptorl.llm.models import NewsArticle


async def test_deepseek_integration():
    """Test DeepSeek API integration."""
    
    # Ensure we have DeepSeek API key
    if not settings.deepseek_api_key or settings.deepseek_api_key == "your_deepseek_api_key_here":
        logger.error("Please set DEEPSEEK_API_KEY in your .env file")
        return False
    
    try:
        # Create DeepSeek client
        llm_client = LLMFactory.create_client(settings)
        
        # Test health check
        health = await llm_client.health_check()
        logger.info(f"Health check: {health}")
        
        if health["status"] != "healthy":
            logger.error("DeepSeek client is not healthy")
            return False
        
        # Create test data
        test_news = [
            NewsArticle(
                title="Bitcoin Reaches New All-Time High Amid Institutional Adoption",
                content="Bitcoin surged to $75,000 today as major corporations announced Bitcoin treasury allocations. MicroStrategy added another 1,000 BTC to its holdings.",
                source="CoinDesk",
                published_date=datetime.now(),
                relevance_score=0.95,
                sentiment_score=0.85
            ),
            NewsArticle(
                title="Federal Reserve Signals Potential Rate Cuts",
                content="The Federal Reserve hinted at potential interest rate cuts in the coming months, boosting risk assets including cryptocurrencies.",
                source="Reuters",
                published_date=datetime.now(),
                relevance_score=0.80,
                sentiment_score=0.75
            )
        ]
        
        test_market_context = {
            'current_price': 75000,
            'price_change_24h': 5.2,
            'volume_24h': 35000000000,
            'market_cap': 1475000000000,
            'volatility_7d': 3.8,
            'previous_close': 71300,
            'high_24h': 75500,
            'low_24h': 71200
        }
        
        # Generate confidence vector
        logger.info("Testing confidence vector generation...")
        response = await llm_client.generate_confidence_vector(
            symbol="BTCUSDT",
            date=datetime.now().strftime('%Y-%m-%d'),
            news_articles=test_news,
            market_context=test_market_context
        )
        
        logger.info(f"Generated confidence vector: {response.confidence_vector}")
        logger.info(f"Model used: {response.model_name}")
        logger.info(f"Processing time: {response.processing_time:.2f}s")
        logger.info(f"Tokens used: {response.prompt_tokens} prompt, {response.completion_tokens} completion")
        logger.info(f"Reasoning: {response.reasoning[:200]}...")
        
        # Test batch generation
        logger.info("Testing batch generation...")
        test_requests = [
            {
                'symbol': 'BTCUSDT',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'news_articles': test_news,
                'market_context': test_market_context
            }
        ]
        
        async for batch_response in llm_client.batch_generate(test_requests):
            logger.info(f"Batch response: {batch_response.confidence_vector}")
            break
        
        # Close client
        if hasattr(llm_client, 'close'):
            await llm_client.close()
        
        logger.success("DeepSeek integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"DeepSeek integration test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_deepseek_integration())