"""LangSearch API client for enhanced web search and semantic reranking."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import httpx
from loguru import logger
import json
import time

from .models import NewsArticle, SearchQuery
from ..config.settings import Settings


class LangSearchClient:
    """Client for LangSearch API with web search and semantic reranking capabilities."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.langsearch_api_key
        self.base_url = "https://api.langsearch.com/v1"
        self._initialized = bool(self.api_key)
        self._rate_limiter = asyncio.Semaphore(5)  # Max 5 concurrent requests
        self._request_times = []
        self._min_request_interval = 0.2  # 200ms between requests
        self._last_request_time = 0
        
    async def initialize(self):
        """Initialize the LangSearch API client."""
        if not self.api_key:
            raise ValueError("LangSearch API key not configured")
        logger.info("Initialized LangSearch API client")
    
    async def _rate_limit(self):
        """Implement rate limiting to prevent API throttling."""
        async with self._rate_limiter:
            now = time.time()
            time_since_last = now - self._last_request_time
            
            if time_since_last < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - time_since_last)
            
            self._last_request_time = time.time()
    
    async def _make_request(self, endpoint: str, payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Make HTTP request with retry logic and rate limiting."""
        await self._rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.base_url}/{endpoint}",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('retry-after', 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s before retry {attempt + 1}")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                logger.error(f"HTTP error in LangSearch API: {e}")
                raise
            except Exception as e:
                logger.error(f"Error in LangSearch API request: {e}")
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded for LangSearch API")
    
    async def search_news(
        self,
        symbol: str,
        date: datetime,
        keywords: List[str] = None,
        max_results: int = 10,
        freshness: str = "oneWeek",
        include_summary: bool = True
    ) -> List[NewsArticle]:
        """Search for news articles using LangSearch web search API."""
        
        if not self._initialized:
            await self.initialize()
        
        # Build search query
        query = SearchQuery(
            symbol=symbol,
            date=date,
            keywords=keywords or [],
            max_results=max_results
        )
        
        search_query = query.to_search_string()
        
        try:
            payload = {
                "query": search_query,
                "freshness": freshness,
                "summary": include_summary,
                "count": max_results
            }
            
            data = await self._make_request("web-search", payload)
            
            articles = []
            if "data" in data and "webPages" in data["data"] and "value" in data["data"]["webPages"]:
                for result in data["data"]["webPages"]["value"]:
                    article = NewsArticle(
                        title=result.get("name", ""),
                        content=result.get("summary", result.get("snippet", "")),
                        source=self._extract_source_from_url(result.get("url", "")),
                        published_date=self._parse_date(result.get("datePublished")),
                        url=result.get("url", ""),
                        relevance_score=self._calculate_relevance_score(result, query)
                    )
                    articles.append(article)
            
            logger.info(f"Found {len(articles)} articles via LangSearch")
            return articles
            
        except Exception as e:
            logger.error(f"LangSearch API error: {e}")
            return []
    
    async def rerank_articles(
        self,
        query: str,
        articles: List[NewsArticle],
        top_n: Optional[int] = None,
        return_documents: bool = True
    ) -> List[NewsArticle]:
        """Rerank articles using semantic relevance with LangSearch rerank API."""
        
        if not self._initialized:
            await self.initialize()
        
        if not articles:
            return []
        
        try:
            # Prepare documents for reranking
            documents = []
            for article in articles:
                document_text = f"{article.title}\n{article.content}"
                documents.append(document_text)
            
            payload = {
                "model": "langsearch-reranker-v1",
                "query": query,
                "documents": documents,
                "top_n": top_n or len(documents),
                "return_documents": return_documents
            }
            
            data = await self._make_request("rerank", payload)
            
            # Create reranked articles
            reranked_articles = []
            if "results" in data:
                for result in data["results"]:
                    index = result.get("index", 0)
                    relevance_score = result.get("relevance_score", 0.5)
                    
                    if 0 <= index < len(articles):
                        article = articles[index]
                        article.relevance_score = relevance_score
                        reranked_articles.append(article)
            
            logger.info(f"Reranked {len(reranked_articles)} articles")
            return reranked_articles
            
        except Exception as e:
            logger.error(f"LangSearch rerank error: {e}")
            return articles  # Return original articles if reranking fails
    
    async def search_and_rerank(
        self,
        symbol: str,
        date: datetime,
        keywords: List[str] = None,
        max_results: int = 10,
        rerank_top_n: Optional[int] = None
    ) -> List[NewsArticle]:
        """Search for news and apply semantic reranking."""
        
        # First, search for articles
        articles = await self.search_news(
            symbol=symbol,
            date=date,
            keywords=keywords,
            max_results=max_results,
            freshness="oneWeek"
        )
        
        if not articles:
            return []
        
        # Then rerank based on semantic relevance
        search_query = f"{symbol} cryptocurrency trading news {date.strftime('%Y-%m-%d')}"
        reranked_articles = await self.rerank_articles(
            query=search_query,
            articles=articles,
            top_n=rerank_top_n or max_results,
            return_documents=True
        )
        
        return reranked_articles
    
    def _extract_source_from_url(self, url: str) -> str:
        """Extract news source from URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain.replace('www.', '').split('.')[0].title()
        except:
            return "Unknown"
    
    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse date string to datetime object."""
        if not date_str:
            return datetime.utcnow()
        
        try:
            # Handle ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return datetime.utcnow()
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: SearchQuery) -> float:
        """Calculate relevance score for search results."""
        score = 0.5  # Base score
        
        # Title relevance
        title = result.get("name", "").lower()
        symbol_lower = query.symbol.lower().replace("usdt", "")
        
        if symbol_lower in title:
            score += 0.3
        
        # Content relevance
        content = result.get("snippet", "").lower()
        if symbol_lower in content:
            score += 0.2
        
        # Date proximity (if available)
        date_published = result.get("datePublished")
        if date_published:
            try:
                pub_date = self._parse_date(date_published)
                days_diff = abs((query.date - pub_date).days)
                if days_diff <= 1:
                    score += 0.2
                elif days_diff <= 3:
                    score += 0.1
            except:
                pass
        
        return min(score, 1.0)
    
    async def search_domain_specific(
        self,
        symbol: str,
        date: datetime,
        domain: str,
        max_results: int = 5
    ) -> List[NewsArticle]:
        """Search for news specific to a confidence vector domain."""
        
        domain_queries = {
            "fundamentals": f"{symbol.replace('USDT', '')} cryptocurrency project fundamentals technology adoption partnerships",
            "industry_condition": "cryptocurrency blockchain industry market trends sector analysis",
            "geopolitics": "global geopolitics cryptocurrency bitcoin regulation international relations",
            "macroeconomics": "global economy macroeconomic indicators federal reserve inflation cryptocurrency impact",
            "technical_sentiment": f"{symbol.replace('USDT', '')} technical analysis market sentiment trading patterns",
            "regulatory_impact": "cryptocurrency regulation SEC CFTC government policy bitcoin ETF approval",
            "innovation_impact": "blockchain innovation cryptocurrency technology development DeFi NFT Web3"
        }
        
        if domain not in domain_queries:
            logger.warning(f"Unknown domain: {domain}")
            return []
        
        query = domain_queries[domain]
        
        try:
            payload = {
                "query": query,
                "freshness": "oneWeek",
                "summary": True,
                "count": max_results
            }
            
            data = await self._make_request("web-search", payload)
            
            articles = []
            if "data" in data and "webPages" in data["data"] and "value" in data["data"]["webPages"]:
                for result in data["data"]["webPages"]["value"]:
                    article = NewsArticle(
                        title=result.get("name", ""),
                        content=result.get("summary", result.get("snippet", "")),
                        source=self._extract_source_from_url(result.get("url", "")),
                        published_date=self._parse_date(result.get("datePublished")),
                        url=result.get("url", ""),
                        relevance_score=0.5,
                        tags=[domain]
                    )
                    articles.append(article)
            
            logger.info(f"Found {len(articles)} articles for domain: {domain}")
            return articles
            
        except Exception as e:
            logger.error(f"Error searching domain {domain}: {e}")
            return []
    
    async def search_all_domains(
        self,
        symbol: str,
        date: datetime,
        max_results_per_domain: int = 3
    ) -> Dict[str, List[NewsArticle]]:
        """Search for news across all 7 confidence vector domains."""
        
        domains = [
            "fundamentals", "industry_condition", "geopolitics",
            "macroeconomics", "technical_sentiment", "regulatory_impact", "innovation_impact"
        ]
        
        results = {}
        
        # Search each domain sequentially to avoid overwhelming the API
        for domain in domains:
            try:
                articles = await self.search_domain_specific(
                    symbol=symbol,
                    date=date,
                    domain=domain,
                    max_results=max_results_per_domain
                )
                results[domain] = articles
                
                # Add small delay between domain searches
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to search domain {domain}: {e}")
                results[domain] = []
        
        return results
    
    async def search_global_context(
        self,
        date: datetime,
        max_results: int = 5
    ) -> List[NewsArticle]:
        """Search for global economic and political news affecting crypto."""
        
        global_queries = [
            "global economy cryptocurrency impact",
            "federal reserve bitcoin price federal interest rates",
            "china cryptocurrency regulation bitcoin mining",
            "european union crypto regulation MiCA",
            "japan cryptocurrency policy digital yen",
            "sec bitcoin ETF approval cryptocurrency regulation",
            "imf world bank cryptocurrency digital currency",
            "biden administration cryptocurrency policy executive order",
            "trump cryptocurrency bitcoin strategic reserve",
            "brics cryptocurrency bitcoin adoption"
        ]
        
        all_articles = []
        
        for query in global_queries[:max_results]:
            try:
                payload = {
                    "query": query,
                    "freshness": "oneWeek",
                    "summary": True,
                    "count": 2
                }
                
                data = await self._make_request("web-search", payload)
                
                if "data" in data and "webPages" in data["data"] and "value" in data["data"]["webPages"]:
                    for result in data["data"]["webPages"]["value"]:
                        article = NewsArticle(
                            title=result.get("name", ""),
                            content=result.get("summary", result.get("snippet", "")),
                            source=self._extract_source_from_url(result.get("url", "")),
                            published_date=self._parse_date(result.get("datePublished")),
                            url=result.get("url", ""),
                            relevance_score=0.7,
                            tags=["global_context"]
                        )
                        all_articles.append(article)
                
                # Delay between global searches
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error in global context search: {e}")
                continue
        
        return all_articles
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LangSearch API client health."""
        try:
            if not self.api_key:
                return {
                    "status": "unhealthy",
                    "error": "LangSearch API key not configured"
                }
            
            # Test with a simple search
            payload = {
                "query": "Bitcoin crypto news",
                "count": 1
            }
            
            await self._make_request("web-search", payload)
            
            return {
                "status": "healthy",
                "initialized": self._initialized,
                "api_key_configured": bool(self.api_key)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }