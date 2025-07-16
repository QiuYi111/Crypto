# China-Compatible Search API Setup Guide (Updated for DuckDuckGo)

This guide explains how to set up China-compatible search APIs for the CryptoRL agent, now using DuckDuckGo as the primary search engine.

## ✅ **DuckDuckGo Search (Primary - No API Key Required)**

DuckDuckGo provides free search capabilities accessible from China without requiring API keys or accounts.

**Features:**
- ✅ **No API key required** - works out of the box
- ✅ **China accessible** - no VPN needed
- ✅ **Privacy-focused** - no tracking
- ✅ **Free unlimited usage**
- ✅ **Real-time news results**

**Installation (optional for enhanced scraping):**
```bash
pip install beautifulsoup4 lxml
```

## 🔍 **Alternative Search APIs**

### 2. Baidu Custom Search API (Optional)
Baidu's search API for Chinese market news.

**Setup:**
1. Go to [Baidu Developer Center](https://ai.baidu.com/)
2. Create a Custom Search Engine
3. Get your API key
4. Set in `.env`:
   ```
   BAIDU_API_KEY=your_baidu_api_key
   ```

### 3. SerpAPI (Fallback)
Good for international news if other sources fail.

**Setup:**
1. Go to [SerpAPI](https://serpapi.com/)
2. Sign up → Copy API key from dashboard
3. Set in `.env`:
   ```
   SERPAPI_KEY=your_serpapi_key_here
   ```

**Cost**: Free tier = 100 searches/month

### 4. Google Custom Search (Requires VPN)
Traditional Google search for comprehensive results.

**Setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create Custom Search Engine: https://cse.google.com/
3. Get API key and Search Engine ID
4. Set in `.env`:
   ```
   GOOGLE_SEARCH_API_KEY=your_google_api_key_here
   GOOGLE_SEARCH_CX=your_custom_search_id_here
   ```

## ⚙️ **Configuration Priority (Updated)**

The system automatically uses this priority order:
1. **DuckDuckGo** (works immediately, no setup)
2. **Baidu** (if API key provided)
3. **SerpAPI** (if API key provided)
4. **Google Search** (if configured)

## 🚀 **Quick Start (No Setup Required)**

**DuckDuckGo works immediately** - no configuration needed:

```bash
# Just run the system - DuckDuckGo will work out of the box
uv run python -m cryptorl.main
```

## 📋 **Environment Variables** (Optional)

Add these to your `.env` file for additional sources:

```bash
# Optional: Baidu for Chinese news
BAIDU_API_KEY=your_baidu_api_key_here

# Optional: SerpAPI for international news
SERPAPI_KEY=your_serpapi_key_here

# Optional: Google Search (requires VPN)
GOOGLE_SEARCH_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_CX=your_custom_search_id_here
```

## 🧪 **Testing**

Run the test script to verify all search sources:

```bash
python test_china_search.py
```

## 🔧 **Troubleshooting**

### DuckDuckGo Issues
- Ensure internet connection is stable
- Try running with verbose logging: `python -c "from src.cryptorl.llm.rag_pipeline import RAGPipeline; ..."`

### Missing Dependencies
If you see BeautifulSoup errors:
```bash
pip install beautifulsoup4 lxml
```

### Network Issues
- Check firewall settings allow HTTPS
- DuckDuckGo should work from most locations in China
- Try accessing https://duckduckgo.com directly in browser

## 📞 **Support**

**DuckDuckGo**: Always available, no support needed
**Baidu**: [Baidu Developer Support](https://ai.baidu.com/support)
**SerpAPI**: [SerpAPI Discord](https://discord.gg/serpapi)

The system now prioritizes DuckDuckGo which works without any setup or API keys!