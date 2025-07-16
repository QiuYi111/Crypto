"""DeepSeek API client for generating confidence vectors from news and market data."""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from loguru import logger

from .models import LLMResponse, NewsArticle, ConfidenceVector
from ..config.settings import Settings


class DeepSeekClient:
    """Client for interacting with DeepSeek API for confidence vector generation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.deepseek_api_key
        self.base_url = "https://api.deepseek.com"
        self.model = settings.deepseek_model or "deepseek-chat"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        self._initialized = bool(self.api_key)
        
    async def initialize(self):
        """Initialize the DeepSeek client."""
        if not self._initialized:
            logger.warning("DeepSeek API key not configured")
            return
            
        try:
            # Test API connection
            response = await self.client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1
                }
            )
            response.raise_for_status()
            logger.info("DeepSeek API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")
            raise
    
    async def generate_confidence_vector(
        self,
        symbol: str,
        date: str,
        news_articles: List[NewsArticle],
        market_context: Dict[str, Any]
    ) -> LLMResponse:
        """Generate confidence vector from news articles and market context using DeepSeek."""
        if not self._initialized:
            raise ValueError("DeepSeek API key not configured")
        
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(symbol, date, news_articles, market_context)
        
        try:
            # Make API request to DeepSeek
            response = await self.client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_tokens,
                    "top_p": self.settings.llm_top_p
                }
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()
            
            # Parse response
            confidence_vector = self._parse_llm_response(response_text)
            
            # Calculate token counts from API response
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                confidence_vector=confidence_vector,
                reasoning=response_text,
                model_name=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error generating confidence vector with DeepSeek: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for DeepSeek."""
        return """You are an expert cryptocurrency market analyst with deep knowledge of blockchain technology, market dynamics, and macroeconomic factors. 

Your task is to analyze news and market data to provide precise, data-driven confidence assessments for cryptocurrency investments. Be analytical, objective, and quantitative in your approach.

Always provide your response in the exact format requested, with numerical scores between 0 and 1."""
    
    def _build_prompt(
        self,
        symbol: str,
        date: str,
        news_articles: List[NewsArticle],
        market_context: Dict[str, Any]
    ) -> str:
        """Build comprehensive prompt for DeepSeek analysis."""
        
        # Format news articles
        news_text = ""
        for i, article in enumerate(news_articles[:5], 1):  # Limit to top 5 articles
            news_text += f"""
{i}. {article.title}
   Source: {article.source}
   Date: {article.published_date.strftime('%Y-%m-%d')}
   Summary: {article.content[:300]}...
   Relevance: {article.relevance_score or 'N/A'}
   Sentiment: {article.sentiment_score or 'N/A'}
"""
        
        # Format market context
        market_info = f"""
Market Context for {symbol} on {date}:
- Current Price: ${market_context.get('current_price', 'N/A')}
- 24h Change: {market_context.get('price_change_24h', 'N/A')}%
- Volume: ${market_context.get('volume_24h', 'N/A'):,}
- Market Cap: ${market_context.get('market_cap', 'N/A'):,}
- Volatility (7d): {market_context.get('volatility_7d', 'N/A')}%
- Previous Close: ${market_context.get('previous_close', 'N/A')}
- High/Low (24h): ${market_context.get('high_24h', 'N/A')} / ${market_context.get('low_24h', 'N/A')}
"""
        
        prompt = f"""Analyze {symbol} for {date} based on the following comprehensive data:

{market_info}

News Analysis:
{news_text}

Based on this information, provide a confidence vector with 7 dimensions:
1. Fundamentals (0-1): Confidence in underlying project fundamentals, team, technology, adoption
2. Industry Condition (0-1): Health and growth prospects of the crypto/blockchain industry
3. Geopolitics (0-1): Impact of geopolitical events, regulations, and international relations
4. Macroeconomics (0-1): Macroeconomic environment impact (inflation, rates, economic growth)
5. Technical Sentiment (0-1): Technical analysis indicators, chart patterns, momentum
6. Regulatory Impact (0-1): Regulatory environment clarity and potential changes
7. Innovation Impact (0-1): Technological innovation, partnerships, development activity

Output format (FIRST LINE ONLY):
[score1,score2,score3,score4,score5,score6,score7]

Then provide a brief 2-3 sentence explanation for the most significant scores (>0.7 or <0.3)."""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> List[float]:
        """Parse DeepSeek response to extract confidence vector."""
        try:
            # Look for JSON array pattern
            import re
            array_pattern = r'\[([\d.,\s\-]+)\]'
            match = re.search(array_pattern, response_text)
            
            if match:
                numbers_str = match.group(1)
                numbers = [float(x.strip()) for x in numbers_str.split(',')]
                
                # Ensure we have 7 values, pad with 0.5 if necessary
                if len(numbers) == 7:
                    # Clamp values to [0, 1]
                    return [max(0, min(1, val)) for val in numbers]
                elif len(numbers) > 7:
                    return [max(0, min(1, val)) for val in numbers[:7]]
                else:
                    padded = numbers + [0.5] * (7 - len(numbers))
                    return [max(0, min(1, val)) for val in padded]
            
            # Fallback: return neutral vector
            logger.warning("Could not parse confidence vector from DeepSeek response")
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            
        except Exception as e:
            logger.error(f"Error parsing DeepSeek response: {e}")
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    async def batch_generate(
        self,
        requests: List[Dict[str, Any]]
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate confidence vectors for multiple requests in batch."""
        if not self._initialized:
            raise ValueError("DeepSeek API key not configured")
            
        # Process in parallel with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def process_single_request(request):
            async with semaphore:
                return await self.generate_confidence_vector(
                    symbol=request['symbol'],
                    date=request['date'],
                    news_articles=request['news_articles'],
                    market_context=request['market_context']
                )
        
        tasks = [process_single_request(request) for request in requests]
        
        for task in asyncio.as_completed(tasks):
            try:
                response = await task
                yield response
            except Exception as e:
                logger.error(f"Error processing batch request: {e}")
                continue
    
    async def health_check(self) -> Dict[str, Any]:
        """Check DeepSeek client health."""
        try:
            if not self._initialized:
                return {
                    "status": "unhealthy",
                    "error": "API key not configured"
                }
            
            # Test API connection
            response = await self.client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1
                }
            )
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "model": self.model,
                "initialized": self._initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()