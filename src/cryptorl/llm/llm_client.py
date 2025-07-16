"""LLM client for generating confidence vectors using DeepSeek API."""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from loguru import logger
import json

from .models import LLMResponse, NewsArticle, ConfidenceVector
from ..config.settings import Settings


class LLMClient:
    """Client for interacting with DeepSeek API for confidence vector generation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
        self.model_name = settings.deepseek_model
        self._initialized = bool(self.api_key)
        
    async def initialize(self):
        """Initialize the DeepSeek API client."""
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")
        logger.info(f"Initialized DeepSeek API client: {self.model_name}")
    
    async def generate_confidence_vector(
        self,
        symbol: str,
        date: str,
        news_articles: List[NewsArticle],
        market_context: Dict[str, Any]
    ) -> LLMResponse:
        """Generate confidence vector using DeepSeek API."""
        if not self._initialized:
            await self.initialize()
            
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(symbol, date, news_articles, market_context)
        
        try:
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency market analyst. Analyze the provided data and return ONLY a JSON response with confidence scores."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
                "top_p": self.settings.llm_top_p,
                "response_format": {"type": "json_object"}
            }
            
            prompt_tokens = len(prompt.split()) * 1.5  # Rough estimate
            
            # Make API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract response
                content = result["choices"][0]["message"]["content"]
                response_data = json.loads(content)
                
                # Parse confidence vector from JSON
                confidence_vector = self._parse_json_response(response_data)
                
                completion_tokens = len(content.split())
                processing_time = time.time() - start_time
                
                return LLMResponse(
                    confidence_vector=confidence_vector,
                    reasoning=content,
                    model_name=self.model_name,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=completion_tokens,
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Error generating confidence vector with DeepSeek: {e}")
            # Return fallback vector
            return LLMResponse(
                confidence_vector=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                reasoning=f"Fallback due to API error: {str(e)}",
                model_name=self.model_name,
                prompt_tokens=100,
                completion_tokens=50,
                processing_time=time.time() - start_time
            )
    
    def _build_prompt(
        self,
        symbol: str,
        date: str,
        news_articles: List[NewsArticle],
        market_context: Dict[str, Any]
    ) -> str:
        """Build comprehensive prompt for DeepSeek API."""
        
        # Format news articles
        news_text = ""
        for i, article in enumerate(news_articles[:5], 1):
            news_text += f"""
{i}. {article.title}
   Source: {article.source}
   Date: {article.published_date.strftime('%Y-%m-%d') if hasattr(article.published_date, 'strftime') else str(article.published_date)}
   Summary: {article.content[:200]}...
   Relevance: {article.relevance_score or 0.5:.2f}
   Sentiment: {article.sentiment_score or 0.0:.2f}
"""
        
        # Format market context (with fallback values)
        market_info = f"""
Market Context for {symbol} on {date}:
- Current Price: ${market_context.get('current_price', 'Unknown')}
- 24h Change: {market_context.get('price_change_24h', 'Unknown')}
- Volume: {market_context.get('volume_24h', 'Unknown')}
- Market Cap: {market_context.get('market_cap', 'Unknown')}
- Volatility (7d): {market_context.get('volatility_7d', 'Unknown')}
"""
        
        prompt = f"""Analyze the following cryptocurrency news and market data for {symbol} on {date}.

{market_info}

News Analysis:
{news_text}

Based on this information, generate a confidence assessment in JSON format with the following structure:

{{"confidence_vector": [fundamentals, industry_condition, geopolitics, macroeconomics, technical_sentiment, regulatory_impact, innovation_impact], "reasoning": "Brief explanation"}}

Where each confidence score is a float between 0.0 and 1.0:
- fundamentals: Confidence in underlying project fundamentals
- industry_condition: Health of the crypto/blockchain industry
- geopolitics: Impact of geopolitical events on this asset
- macroeconomics: Macroeconomic environment impact
- technical_sentiment: Technical analysis sentiment
- regulatory_impact: Regulatory environment impact
- innovation_impact: Technological innovation impact

Return only valid JSON."""
        
        return prompt
    
    def _parse_json_response(self, response_data: Dict[str, Any]) -> List[float]:
        """Parse confidence vector from JSON response."""
        try:
            # Try to get vector from response
            if "confidence_vector" in response_data:
                vector = response_data["confidence_vector"]
            elif "vector" in response_data:
                vector = response_data["vector"]
            else:
                # Try to extract array from any key
                for key, value in response_data.items():
                    if isinstance(value, list) and len(value) == 7:
                        vector = value
                        break
                else:
                    raise ValueError("No confidence vector found in response")
            
            # Ensure we have 7 float values between 0 and 1
            if len(vector) == 7:
                return [max(0.0, min(1.0, float(v))) for v in vector]
            else:
                # Pad or truncate to 7 values
                padded = list(vector)[:7] + [0.5] * (7 - min(7, len(vector)))
                return [max(0.0, min(1.0, float(v))) for v in padded]
                
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    async def batch_generate(
        self,
        requests: List[Dict[str, Any]]
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate confidence vectors for multiple requests in batch."""
        for request in requests:
            try:
                response = await self.generate_confidence_vector(
                    symbol=request['symbol'],
                    date=request['date'],
                    news_articles=request['news_articles'],
                    market_context=request['market_context']
                )
                yield response
            except Exception as e:
                logger.error(f"Error processing request {request}: {e}")
                continue
    
    async def health_check(self) -> Dict[str, Any]:
        """Check DeepSeek API client health."""
        try:
            if not self.api_key:
                return {
                    "status": "unhealthy",
                    "error": "DeepSeek API key not configured"
                }
            
            # Simple test call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 1
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "initialized": self._initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }