"""LangSearch API client for enhanced web search and semantic reranking."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger
import json

from .models import NewsArticle, SearchQuery
from ..config.settings import Settings


class LangSearchClient:
    """Client for LangSearch API with web search and semantic reranking capabilities."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.langsearch_api_key
        self.base_url = "https://api.langsearch.com/v1"
        self._initialized = bool(self.api_key)
        
    async def initialize(self):
        """Initialize the LangSearch API client."""
        if not self.api_key:
            raise ValueError("LangSearch API key not configured")
        logger.info("Initialized LangSearch API client")
    
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
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": search_query,
                "freshness": freshness,
                "summary": include_summary,
                "count": max_results
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/web-search",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
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
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
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
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/rerank",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
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
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LangSearch API client health."""
        try:
            if not self.api_key:
                return {
                    "status": "unhealthy",
                    "error": "LangSearch API key not configured"
                }
            
            # Test with a simple search
            test_query = "Bitcoin crypto news"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": test_query,
                "count": 1
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/web-search",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
            
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