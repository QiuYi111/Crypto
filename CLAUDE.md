# CryptoRL Agent

## What This Does
A crypto day-trading bot that uses AI to read news and make trading decisions. It combines:
- **Reinforcement Learning**: Learns when to buy/sell crypto
- **LLM Analysis**: Reads news to understand market sentiment
- **Mamba Architecture**: Advanced neural network for time series

## Data Flow (How Parts Connect)

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
│  Binance API → InfluxDB → Data Fusion → RL Agent → Trading │
│        ↓              ↓          ↓           ↓      ↓       │
│     OHLCV Data → Storage → + LLM Vectors → Decisions → Orders │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step Process
1. **Collect**: Binance API fetches BTC/ETH/SOL price data every hour
2. **Store**: Saves to InfluxDB (time-series) and PostgreSQL (trades)
3. **Analyze**: LLM reads crypto news → generates 7D confidence vector
4. **Fuse**: Combines price data + confidence vectors
5. **Train**: RL agent learns optimal trading strategy
6. **Trade**: Executes buy/sell orders on Binance

## Confidence Vector Pipeline
```
News Articles → LLM Analysis → [0.7, 0.6, 0.4, 0.8, 0.5, 0.3, 0.6]
```
**What each number means:**
- Fundamentals: Company/project health
- Industry: Overall crypto market sentiment  
- Geopolitics: Regulatory/political events
- Macroeconomics: Economic indicators
- Technical: Price chart patterns
- Regulatory: Government policy changes
- Innovation: New tech developments

## Quick Commands
```bash
# Setup everything
python quickstart.py --setup

# See if it works
python scripts/validate_setup.py

# Start monitoring dashboard
python quickstart.py --dashboard
```

## Code Structure (Where to Find Things)
```
src/cryptorl/
├── data/           # Talks to Binance API & databases
├── llm/            # Generates confidence vectors from news
├── rl/             # The AI brain (Mamba models + training)
├── trading/        # Places actual orders on Binance
├── monitoring/     # Shows pretty charts in browser
└── config/         # All settings in one place
```

## Environment Setup
```bash
# Copy this file and fill in your keys
cp .env.example .env

# Required variables
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
INFLUXDB_TOKEN=your_influx_token
DEEPSEEK_API_KEY=your_deepseek_key  # or OPENAI_API_KEY
```

## Key Files for New Interns
- `config/settings.py` - All configuration (start here)
- `quickstart.py` - Master setup script
- `validate_setup.py` - Tests if everything works
- `scripts/phase3_demo.py` - Shows Mamba vs regular models
- `scripts/phase4_demo.py` - Shows backtesting results

## Current Status
- ✅ All 5 phases complete - 95% working
- ⚠️ Just needs API keys in .env file
- ✅ Ready for testing and improvements