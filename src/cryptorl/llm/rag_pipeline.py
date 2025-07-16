"""RAG (Retrieval-Augmented Generation) pipeline for historical news retrieval."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger
import json
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .models import NewsArticle, SearchQuery
from ..config.settings import Settings


class RAGPipeline:
    """Pipeline for retrieving relevant news articles using search APIs."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.serpapi_key = settings.serpapi_key
        self.google_search_key = settings.google_search_api_key
        self.google_search_cx = settings.google_search_cx
        self.baidu_api_key = settings.baidu_api_key
        
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
        
        # Try multiple search sources in priority order
        articles = []
        
        # SerpAPI search (primary choice)
        if self.serpapi_key:
            serp_articles = await self._search_serpapi(query)
            articles.extend(serp_articles)
            if serp_articles:
                logger.info(f"Found {len(serp_articles)} articles via SerpAPI")
        
        # Google Custom Search (secondary choice)
        if self.google_search_key and self.google_search_cx and not articles:
            google_articles = await self._search_google(query)
            articles.extend(google_articles)
            if google_articles:
                logger.info(f"Found {len(google_articles)} articles via Google Search")
        
        # DuckDuckGo search (China-compatible fallback)
        if not articles:
            duckduckgo_articles = await self._search_duckduckgo(query)
            articles.extend(duckduckgo_articles)
            if duckduckgo_articles:
                logger.info(f"Found {len(duckduckgo_articles)} articles via DuckDuckGo")
        
        # Baidu Search (China-specific final fallback)
        if self.baidu_api_key and not articles:
            baidu_articles = await self._search_baidu(query)
            articles.extend(baidu_articles)
            if baidu_articles:
                logger.info(f"Found {len(baidu_articles)} articles via Baidu")
        
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
            
            # Format date range (±3 days around target date)
            date_start = query.date - timedelta(days=3)
            date_end = query.date + timedelta(days=3)
            
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
            
            async with httpx.AsyncClient(timeout=30.0, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }) as client:
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
            logger.debug(f"Query: {query.to_search_string()}, Date: {query.date}")
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
            
            async with httpx.AsyncClient(timeout=30.0, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }) as client:
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
    
    async def _search_duckduckgo(self, query: SearchQuery) -> List[NewsArticle]:
        """Search using DuckDuckGo API (no API key required)."""
        try:
            # Use DuckDuckGo's instant answer API
            search_url = "https://api.duckduckgo.com/"
            
            params = {
                "q": query.to_search_string(),
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
                "t": "cryptorl-agent"
            }
            
            async with httpx.AsyncClient(timeout=30.0, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }) as client:
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                articles = []
                
                # Process news results
                for result in data.get("RelatedTopics", [])[:query.max_results]:
                    if isinstance(result, dict) and "Text" in result:
                        article = NewsArticle(
                            title=result.get("Text", "")[:100],
                            content=result.get("Text", ""),
                            source="DuckDuckGo",
                            published_date=datetime.utcnow(),  # DuckDuckGo doesn't provide dates
                            url=result.get("FirstURL", ""),
                            relevance_score=self._calculate_relevance_score(result, query)
                        )
                        articles.append(article)
                
                # Also try web scraping approach for more news
                news_articles = await self._scrape_duckduckgo_news(query)
                articles.extend(news_articles)
                
                return articles
                
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            # Fallback to web scraping
            return await self._scrape_duckduckgo_news(query)
    
    async def _scrape_duckduckgo_news(self, query: SearchQuery) -> List[NewsArticle]:
        """Scrape DuckDuckGo news search results."""
        try:
            # Use DuckDuckGo news search
            search_url = "https://duckduckgo.com/html/"
            
            params = {
                "q": query.to_search_string(),
                "t": "h_",
                "ia": "news"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with httpx.AsyncClient(timeout=30.0, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }) as client:
                response = await client.get(search_url, params=params, headers=headers)
                response.raise_for_status()
                
                # Simple HTML parsing for news results
                import re
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = []
                
                # Find news results
                results = soup.find_all('div', class_='result')[:query.max_results]
                
                for result in results:
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = result.find('a', class_='result__snippet')
                    url_elem = result.find('a', class_='result__a')
                    
                    if title_elem and snippet_elem:
                        article = NewsArticle(
                            title=title_elem.get_text(strip=True),
                            content=snippet_elem.get_text(strip=True),
                            source="DuckDuckGo",
                            published_date=datetime.utcnow(),
                            url=url_elem.get('href', '') if url_elem else '',
                            relevance_score=0.7  # Default score for scraped content
                        )
                        articles.append(article)
                
                return articles
                
        except Exception as e:
            logger.error(f"DuckDuckGo scraping failed: {e}")
            return []
    
    
    async def _search_baidu(self, query: SearchQuery) -> List[NewsArticle]:
        """Search using Baidu Custom Search API."""
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            
            # Use Baidu-specific search engine ID
            params = {
                "key": self.baidu_api_key,
                "cx": "017576662512468239146:omuauf_lfve",  # Baidu news search engine
                "q": query.to_search_string(),
                "num": min(query.max_results, 10),
                "sort": "date",
                "dateRestrict": "d7",  # Last 7 days
                "siteSearch": "baidu.com"
            }
            
            async with httpx.AsyncClient(timeout=30.0, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }) as client:
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
            logger.error(f"Baidu search failed: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check RAG pipeline health."""
        try:
            # Test with a simple search
            test_date = datetime.utcnow() - timedelta(days=7)
            test_articles = await self.search_news("BTCUSDT", test_date, max_results=1)
            
            return {
                "status": "healthy",
                "duckduckgo_enabled": True,
                "baidu_search_enabled": bool(self.baidu_api_key),
                "serpapi_enabled": bool(self.serpapi_key),
                "google_search_enabled": bool(self.google_search_key and self.google_search_cx),
                "test_results": len(test_articles)>0,
                "beautifulsoup_available": BeautifulSoup is not None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }