"""RAG (Retrieval-Augmented Generation) pipeline for historical news retrieval."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger
import json

from .models import NewsArticle, SearchQuery
from ..config.settings import Settings


class RAGPipeline:
    """Pipeline for retrieving relevant news articles using search APIs."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.serpapi_key = settings.serpapi_key
        self.google_search_key = settings.google_api_key
        self.google_search_cx = settings.google_cx
        
    async def search_news(
        self,
        symbol: str,
        date: datetime,
        keywords: List[str] = None,
        max_results: int = 10
    ) -> List[NewsArticle]:
        """Search for news articles about a symbol on a specific date."""
        
        query = SearchQuery(
            symbol=symbol,
            date=date,
            keywords=keywords or [],
            max_results=max_results
        )
        
        # Try multiple search sources
        articles = []
        
        # SerpAPI search
        if self.serpapi_key:
            serp_articles = await self._search_serpapi(query)
            articles.extend(serp_articles)
        
        # Google Custom Search
        if self.google_search_key and self.google_search_cx:
            google_articles = await self._search_google(query)
            articles.extend(google_articles)
        
        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article.url and article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
            elif not article.url:  # Include articles without URLs
                unique_articles.append(article)
        
        # Sort by relevance and date
        unique_articles.sort(
            key=lambda x: (x.relevance_score or 0, x.published_date),
            reverse=True
        )
        
        return unique_articles[:max_results]
    
    async def _search_serpapi(self, query: SearchQuery) -> List[NewsArticle]:
        """Search using SerpAPI."""
        try:
            search_url = "https://serpapi.com/search"
            
            # Format date range: search news up to and including the target date
            # This ensures we use all available information up to July 15th
            # to guide July 16th trading decisions
            date_start = query.date - timedelta(days=7)  # Look back 7 days
            date_end = query.date                        # Include target date
            
            params = {
                "engine": "google",
                "q": query.to_search_string(),
                "location": "United States",
                "google_domain": "google.com",
                "gl": "us",
                "hl": "en",
                "tbm": "nws",
                "num": query.max_results,
                "api_key": self.serpapi_key,
                "tbs": f"cdr:1,cd_min:{date_start.strftime('%m/%d/%Y')},cd_max:{date_end.strftime('%m/%d/%Y')}"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                articles = []
                for result in data.get("news_results", []):
                    article = NewsArticle(
                        title=result.get("title", ""),
                        content=result.get("snippet", ""),
                        source=result.get("source", ""),
                        published_date=self._parse_date(result.get("date", "")),
                        url=result.get("link", ""),
                        relevance_score=self._calculate_relevance_score(result, query)
                    )
                    articles.append(article)
                
                return articles
                
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []
    
    async def _search_google(self, query: SearchQuery) -> List[NewsArticle]:
        """Search using Google Custom Search API."""
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            
            # Format date for Google search
            date_start = query.date - timedelta(days=3)
            date_end = query.date + timedelta(days=3)
            
            params = {
                "key": self.google_search_key,
                "cx": self.google_search_cx,
                "q": query.to_search_string(),
                "dateRestrict": f"d{6}",  # Last 6 days
                "num": min(query.max_results, 10),
                "sort": "date:r:high",
                "siteSearch": "coindesk.com,cointelegraph.com,decrypt.co,blockworks.co"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                articles = []
                for item in data.get("items", []):
                    article = NewsArticle(
                        title=item.get("title", ""),
                        content=item.get("snippet", ""),
                        source=self._extract_source_from_url(item.get("link", "")),
                        published_date=self._parse_date(item.get("snippet", "")),
                        url=item.get("link", ""),
                        relevance_score=self._calculate_relevance_score(item, query)
                    )
                    articles.append(article)
                
                return articles
                
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    async def batch_search(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[List[NewsArticle]]:
        """Search for multiple symbols/dates in batch."""
        
        tasks = []
        for request in requests:
            task = self.search_news(
                symbol=request['symbol'],
                date=request['date'],
                keywords=request.get('keywords', []),
                max_results=request.get('max_results', 10)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch search error: {result}")
                final_results.append([])
            else:
                final_results.append(result)
        
        return final_results
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats."""
        try:
            # Try parsing relative dates like "2 days ago"
            if "ago" in date_str.lower():
                days_ago = int(''.join(filter(str.isdigit, date_str)))
                return datetime.utcnow() - timedelta(days=days_ago)
            
            # Try standard formats
            formats = [
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%b %d, %Y",
                "%d %b %Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # Default to today
            return datetime.utcnow()
            
        except Exception:
            return datetime.utcnow()
    
    def _extract_source_from_url(self, url: str) -> str:
        """Extract news source from URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain.replace('www.', '').split('.')[0].title()
        except:
            return "Unknown"
    
    def _calculate_relevance_score(
        self, 
        result: Dict[str, Any], 
        query: SearchQuery
    ) -> float:
        """Calculate relevance score for a search result."""
        score = 0.5  # Base score
        
        # Title relevance
        title = result.get("title", "").lower()
        symbol_lower = query.symbol.lower().replace("usdt", "")
        
        if symbol_lower in title:
            score += 0.3
        
        # Content relevance
        content = result.get("snippet", "").lower()
        if symbol_lower in content:
            score += 0.2
        
        # Date proximity
        if "date" in result:
            article_date = self._parse_date(result["date"])
            days_diff = abs((query.date - article_date).days)
            if days_diff <= 1:
                score += 0.2
            elif days_diff <= 3:
                score += 0.1
        
        return min(score, 1.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check RAG pipeline health."""
        try:
            # Test with a simple search
            test_date = datetime.utcnow() - timedelta(days=7)
            test_articles = await self.search_news("BTCUSDT", test_date, max_results=1)
            
            return {
                "status": "healthy",
                "serpapi_enabled": bool(self.serpapi_key),
                "google_search_enabled": bool(self.google_search_key and self.google_search_cx),
                "test_results": len(test_articles)>0
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }