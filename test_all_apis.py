#!/usr/bin/env python3
"""Comprehensive API test script for CryptoRL - Binance, DeepSeek, and SERPAPI."""

import asyncio
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APITester:
    """Test all external APIs used by CryptoRL."""
    
    def __init__(self):
        self.results = {}
    
    async def test_binance_api(self):
        """Test Binance API connectivity."""
        print("🔍 Testing Binance API...")
        
        api_key = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_SECRET_KEY")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        if not api_key or not secret_key:
            return {"status": "failed", "error": "Missing API keys in .env"}
        
        try:
            # Test Binance testnet API
            if testnet:
                base_url = "https://testnet.binance.vision/api/v3"
            else:
                base_url = "https://api.binance.com/api/v3"
            
            # Test server time endpoint
            response = requests.get(f"{base_url}/time", timeout=10)
            
            if response.status_code == 200:
                server_time = response.json()["serverTime"]
                
                # Test ticker endpoint
                ticker_response = requests.get(f"{base_url}/ticker/24hr?symbol=BTCUSDT", timeout=10)
                
                if ticker_response.status_code == 200:
                    ticker = ticker_response.json()
                    return {
                        "status": "success",
                        "server_time": server_time,
                        "btc_price": float(ticker["lastPrice"]),
                        "btc_volume": float(ticker["quoteVolume"]),
                        "testnet": testnet,
                        "api_key_found": bool(api_key),
                        "secret_key_found": bool(secret_key)
                    }
                else:
                    return {"status": "failed", "error": f"Ticker API failed: {ticker_response.status_code}"}
            else:
                return {"status": "failed", "error": f"Server time API failed: {response.status_code}"}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_serpapi(self):
        """Test SERPAPI connectivity."""
        print("🔍 Testing SERPAPI...")
        
        api_key = os.getenv("SERPAPI_KEY")
        
        if not api_key:
            return {"status": "failed", "error": "SERPAPI_KEY not found in .env"}
        
        try:
            params = {
                "engine": "google",
                "q": "Bitcoin BTC cryptocurrency news",
                "tbm": "nws",
                "num": 5,
                "api_key": api_key
            }
            
            response = requests.get("https://serpapi.com/search", params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    return {"status": "failed", "error": data["error"]}
                
                news_results = data.get("news_results", [])
                
                return {
                    "status": "success",
                    "articles_found": len(news_results),
                    "api_key_found": True,
                    "sample_articles": [
                        {
                            "title": article.get("title", "No title"),
                            "source": article.get("source", "Unknown"),
                            "link": article.get("link", "No link")
                        }
                        for article in news_results[:3]
                    ]
                }
            else:
                return {"status": "failed", "error": f"HTTP {response.status_code}: {response.text[:200]}"}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_deepseek_api(self):
        """Test DeepSeek API connectivity."""
        print("🔍 Testing DeepSeek API...")
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        if not api_key:
            return {"status": "failed", "error": "DEEPSEEK_API_KEY not found in .env"}
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency analyst. Return a simple JSON response."
                    },
                    {
                        "role": "user",
                        "content": "Analyze Bitcoin sentiment for today in JSON format with confidence scores for fundamentals, industry, geopolitics, macroeconomics, technical, regulatory, and innovation factors (0-1 scale)."
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                return {
                    "status": "success",
                    "model": model,
                    "base_url": base_url,
                    "api_key_found": True,
                    "response": content[:500] + "..." if len(content) > 500 else content,
                    "usage": data.get("usage", {})
                }
            else:
                return {"status": "failed", "error": f"HTTP {response.status_code}: {response.text[:200]}"}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_duckduckgo(self):
        """Test DuckDuckGo search functionality."""
        print("🔍 Testing DuckDuckGo search...")
        
        try:
            import duckduckgo_search
            
            # Test basic search
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                # Test crypto news search
                results = list(ddgs.news(
                    keywords="Bitcoin BTC cryptocurrency",
                    region="us-en",
                    safesearch="Off",
                    timelimit="w",
                    max_results=5
                ))
            
            return {
                "status": "success",
                "articles_found": len(results),
                "sample_articles": [
                    {
                        "title": result.get("title", "No title"),
                        "source": result.get("source", "Unknown"),
                        "url": result.get("url", "No URL"),
                        "date": result.get("date", "No date")
                    }
                    for result in results[:3]
                ]
            }
                
        except ImportError as e:
            return {"status": "failed", "error": f"duckduckgo_search package not installed: {e}"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_all_apis(self):
        """Test all APIs and report results."""
        print("🚀 Starting CryptoRL API Tests...")
        print("=" * 60)
        
        # Test Binance API
        binance_result = await self.test_binance_api()
        self.results["binance"] = binance_result
        
        # Test SERPAPI
        serpapi_result = await self.test_serpapi()
        self.results["serpapi"] = serpapi_result
        
        # Test DeepSeek API
        deepseek_result = await self.test_deepseek_api()
        self.results["deepseek"] = deepseek_result
        
        # Test DuckDuckGo
        duckduckgo_result = await self.test_duckduckgo()
        self.results["duckduckgo"] = duckduckgo_result
        
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 API TEST SUMMARY")
        print("=" * 60)
        
        for api_name, result in self.results.items():
            status = "✅ WORKING" if result["status"] == "success" else "❌ FAILED"
            print(f"\n{api_name.upper()} API: {status}")
            
            if result["status"] == "success":
                for key, value in result.items():
                    if key != "status":
                        print(f"   {key}: {value}")
            else:
                print(f"   Error: {result['error']}")
        
        # Overall status
        working_apis = [k for k, v in self.results.items() if v["status"] == "success"]
        failed_apis = [k for k, v in self.results.items() if v["status"] == "failed"]
        
        print(f"\n📈 SUMMARY:")
        print(f"   Working: {len(working_apis)} ({', '.join(working_apis) if working_apis else 'None'})")
        print(f"   Failed: {len(failed_apis)} ({', '.join(failed_apis) if failed_apis else 'None'})")
        
        # Environment check
        env_vars = ["BINANCE_API_KEY", "BINANCE_SECRET_KEY", "SERPAPI_KEY", "DEEPSEEK_API_KEY"]
        print(f"\n🔑 ENVIRONMENT VARIABLES:")
        for var in env_vars:
            status = "✅ Found" if os.getenv(var) else "❌ Missing"
            print(f"   {var}: {status}")
        
        # DuckDuckGo package check
        try:
            import duckduckgo_search
            print(f"   duckduckgo_search package: ✅ Installed")
        except ImportError:
            print(f"   duckduckgo_search package: ❌ Not installed")

async def main():
    """Main test function."""
    print("🧪 CryptoRL API Comprehensive Test")
    print("Testing: Binance API, SERPAPI, DeepSeek API")
    
    tester = APITester()
    results = await tester.test_all_apis()
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())