"""Main confidence vector generation system orchestrating LLM and RAG components."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import pandas as pd

from .llm_factory import LLMFactory
from .rag_pipeline import RAGPipeline
from .models import ConfidenceVector, NewsArticle
from ..config.settings import Settings
from ..data.influxdb_client import InfluxDBClient


class ConfidenceVectorGenerator:
    """Main system for generating confidence vectors using LLM + RAG pipeline."""
    
    def __init__(self, settings: Settings, influx_client: InfluxDBClient):
        self.settings = settings
        self.llm_client = LLMFactory.create_client(settings)
        self.rag_pipeline = RAGPipeline(settings)
        self.influx = influx_client
        
    async def generate_daily_confidence(
        self,
        symbol: str,
        date: datetime,
        use_cached: bool = True
    ) -> Optional[ConfidenceVector]:
        """Generate confidence vector for a specific symbol and date."""
        
        # Check if already cached
        if use_cached:
            cached = await self._get_cached_vector(symbol, date)
            if cached:
                logger.info(f"Using cached confidence vector for {symbol} on {date.strftime('%Y-%m-%d')}")
                return cached
        
        try:
            # Search for relevant news
            logger.info(f"Searching news for {symbol} on {date.strftime('%Y-%m-%d')}")
            news_articles = await self.rag_pipeline.search_news(
                symbol=symbol,
                date=date,
                max_results=self.settings.llm_max_news_articles
            )
            
            if not news_articles:
                logger.warning(f"No news found for {symbol} on {date.strftime('%Y-%m-%d')}")
                return None
            
            # Get market context
            market_context = await self._get_market_context(symbol, date)
            
            # Generate confidence vector using LLM
            logger.info(f"Generating confidence vector using LLM for {symbol}")
            llm_response = await self.llm_client.generate_confidence_vector(
                symbol=symbol,
                date=date.strftime('%Y-%m-%d'),
                news_articles=news_articles,
                market_context=market_context
            )
            
            # Create confidence vector
            confidence_vector = ConfidenceVector.from_array(
                symbol=symbol,
                date=date,
                vector=llm_response.confidence_vector,
                news_sources=[article.source for article in news_articles],
                confidence_score=0.8,  # Could be calculated from LLM confidence
                reasoning=llm_response.reasoning,
                fundamentals=llm_response.confidence_vector[0],
                industry_condition=llm_response.confidence_vector[1],
                geopolitics=llm_response.confidence_vector[2],
                macroeconomics=llm_response.confidence_vector[3],
                technical_sentiment=llm_response.confidence_vector[4],
                regulatory_impact=llm_response.confidence_vector[5],
                innovation_impact=llm_response.confidence_vector[6]
            )
            
            # Cache the result
            await self._cache_vector(confidence_vector)
            
            return confidence_vector
            
        except Exception as e:
            logger.error(f"Error generating confidence vector for {symbol} on {date}: {e}")
            return None
    
    async def batch_generate_historical(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 10
    ) -> List[ConfidenceVector]:
        """Generate confidence vectors for historical data in batch."""
        
        logger.info(f"Starting batch generation for {len(symbols)} symbols from {start_date} to {end_date}")
        
        all_vectors = []
        
        # Generate date range
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # Create all requests
        requests = []
        for symbol in symbols:
            for date in date_range:
                requests.append({
                    'symbol': symbol,
                    'date': date
                })
        
        # Process in batches
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(requests) + batch_size - 1)//batch_size}")
            
            # Process batch
            tasks = [
                self.generate_daily_confidence(
                    symbol=req['symbol'],
                    date=req['date'],
                    use_cached=True
                )
                for req in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            for result in batch_results:
                if isinstance(result, ConfidenceVector):
                    all_vectors.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
            
            # Rate limiting
            await asyncio.sleep(1)
        
        logger.info(f"Generated {len(all_vectors)} confidence vectors")
        return all_vectors
    
    async def _get_market_context(self, symbol: str, date: datetime) -> Dict[str, Any]:
        """Get market context for a symbol on a specific date."""
        try:
            # Get price data from the previous day
            prev_date = date - timedelta(days=1)
            
            # Query InfluxDB for market data
            price_data = await self.influx.get_price_data(symbol, prev_date)
            
            if price_data.empty:
                return {
                    'current_price': None,
                    'price_change_24h': None,
                    'volume_24h': None,
                    'market_cap': None,
                    'volatility_7d': None
                }
            
            # Calculate context metrics
            latest_price = price_data['close'].iloc[-1]
            prev_price = price_data['close'].iloc[0] if len(price_data) > 1 else latest_price
            price_change = ((latest_price - prev_price) / prev_price) * 100
            
            # Calculate 7-day volatility
            if len(price_data) >= 7:
                volatility = price_data['close'].pct_change().iloc[-7:].std() * 100
            else:
                volatility = price_data['close'].pct_change().std() * 100
            
            return {
                'current_price': latest_price,
                'price_change_24h': price_change,
                'volume_24h': price_data['volume'].iloc[-1],
                'market_cap': latest_price * 21000000,  # Approximate for BTC
                'volatility_7d': volatility
            }
            
        except Exception as e:
            logger.error(f"Error getting market context for {symbol} on {date}: {e}")
            return {}
    
    async def _get_cached_vector(self, symbol: str, date: datetime) -> Optional[ConfidenceVector]:
        """Get cached confidence vector from database."""
        try:
            # Query PostgreSQL for cached vector
            from sqlalchemy import select
            from sqlalchemy.orm import sessionmaker
            from ..data.models import ConfidenceVectorModel
            
            # This would query your PostgreSQL database
            # For now, return None to indicate no cache
            return None
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None
    
    async def _cache_vector(self, vector: ConfidenceVector):
        """Cache confidence vector to database."""
        try:
            # Store in PostgreSQL
            # This would insert into your PostgreSQL database
            logger.debug(f"Caching confidence vector for {vector.symbol} on {vector.date}")
            
        except Exception as e:
            logger.error(f"Error caching vector: {e}")
    
    async def get_confidence_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[ConfidenceVector]:
        """Get historical confidence vectors for a symbol."""
        try:
            # Query cached vectors
            # This would query from PostgreSQL
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting confidence history: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            llm_health = await self.llm_client.health_check()
            rag_health = await self.rag_pipeline.health_check()
            
            return {
                "status": "healthy" if llm_health["status"] == "healthy" and rag_health["status"] == "healthy" else "degraded",
                "llm": llm_health,
                "rag": rag_health
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }