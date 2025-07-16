"""LLM client for generating confidence vectors from news and market data."""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .models import LLMResponse, NewsArticle, ConfidenceVector
from ..config.settings import Settings


class LLMClient:
    """Client for interacting with LLM models for confidence vector generation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.llm.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the LLM model and tokenizer."""
        if self._initialized:
            return
            
        logger.info(f"Initializing LLM: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if configured
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if settings.llm.load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            self._initialized = True
            logger.info(f"LLM initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def generate_confidence_vector(
        self,
        symbol: str,
        date: str,
        news_articles: List[NewsArticle],
        market_context: Dict[str, Any]
    ) -> LLMResponse:
        """Generate confidence vector from news articles and market context."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(symbol, date, news_articles, market_context)
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=self.settings.llm.temperature,
                    top_p=self.settings.llm.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0][len(inputs[0]):], 
                skip_special_tokens=True
            ).strip()
            
            # Parse response
            confidence_vector = self._parse_llm_response(response_text)
            
            # Calculate token counts
            prompt_tokens = len(inputs[0])
            completion_tokens = len(outputs[0]) - len(inputs[0])
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                confidence_vector=confidence_vector,
                reasoning=response_text,
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error generating confidence vector: {e}")
            raise
    
    def _build_prompt(
        self,
        symbol: str,
        date: str,
        news_articles: List[NewsArticle],
        market_context: Dict[str, Any]
    ) -> str:
        """Build comprehensive prompt for LLM analysis."""
        
        # Format news articles
        news_text = ""
        for i, article in enumerate(news_articles[:5], 1):  # Limit to top 5 articles
            news_text += f"""
{i}. {article.title}
   Source: {article.source}
   Date: {article.published_date.strftime('%Y-%m-%d')}
   Summary: {article.content[:200]}...
   Relevance: {article.relevance_score or 'N/A'}
   Sentiment: {article.sentiment_score or 'N/A'}
"""
        
        # Format market context
        market_info = f"""
Market Context for {symbol} on {date}:
- Current Price: ${market_context.get('current_price', 'N/A')}
- 24h Change: {market_context.get('price_change_24h', 'N/A')}
- Volume: {market_context.get('volume_24h', 'N/A')}
- Market Cap: {market_context.get('market_cap', 'N/A')}
- Volatility (7d): {market_context.get('volatility_7d', 'N/A')}
"""
        
        prompt = f"""You are an expert cryptocurrency market analyst. Analyze the provided news and market data to generate a confidence assessment for {symbol} on {date}.

{market_info}

News Analysis:
{news_text}

Based on this information, provide a confidence vector with 7 dimensions:
1. Fundamentals (0-1): Confidence in underlying project fundamentals
2. Industry Condition (0-1): Health of the crypto/blockchain industry
3. Geopolitics (0-1): Impact of geopolitical events on this asset
4. Macroeconomics (0-1): Macroeconomic environment impact
5. Technical Sentiment (0-1): Technical analysis sentiment
6. Regulatory Impact (0-1): Regulatory environment impact
7. Innovation Impact (0-1): Technological innovation impact

Output format:
[score1,score2,score3,score4,score5,score6,score7]

Then provide a brief explanation (2-3 sentences) for each score.

Response:"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> List[float]:
        """Parse LLM response to extract confidence vector."""
        try:
            # Look for JSON array pattern
            import re
            array_pattern = r'\[([\d.,\s]+)\]'
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
            logger.warning("Could not parse confidence vector from LLM response")
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
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
        """Check LLM client health."""
        try:
            if not self._initialized:
                await self.initialize()
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "device": self.device,
                "initialized": self._initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }