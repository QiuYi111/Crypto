# 🚀 CryptoRL Quick Start Guide

Get up and running in 5 minutes with our optimized setup process.

## 📋 Prerequisites

- **Python 3.9+** (tested on 3.9, 3.10, 3.11)
- **Docker** (optional, for database services)
- **Binance API Key** (testnet recommended for development)

## 🎯 5-Minute Setup

### 1. Install Dependencies

```bash
# Install uv (fastest package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

### 2. One-Command Setup

```bash
# Clone repository
git clone <your-repo-url> cryptorl
cd cryptorl

# Automated setup
python quickstart.py --setup
```

### 3. Configure API Keys

```bash
# Edit configuration
nano .env

# Add your keys:
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Use testnet for safe testing
BINANCE_TESTNET=true
```

### 4. Verify Installation

```bash
# Run comprehensive tests
python quickstart.py --test

# Check configuration
python quickstart.py --validate

# Expected output:
# ✅ Python version: 3.11.0
# ✅ Dependencies installed
# ✅ Configuration valid
# ✅ Binance connection successful
```

## 🎛️ Launch Options

### Option A: Dashboard (Recommended)
```bash
# Start interactive dashboard
python quickstart.py --dashboard

# Or direct start
python run_dashboard.py

# Access: http://localhost:8501
```

### Option B: Docker Environment
```bash
# Start full stack
python quickstart.py --docker-up

# Services available:
# - Dashboard: http://localhost:8501
# - InfluxDB: http://localhost:8086
# - PostgreSQL: localhost:5432
```

### Option C: Manual Control
```bash
# Start services individually
uv run streamlit run simple_dashboard.py
```

## 🔧 Configuration Guide

### Environment Variables (.env)
```bash
# Required
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret
DEEPSEEK_API_KEY=your_deepseek_key

# LLM Provider
LLM_PROVIDER=deepseek                 # deepseek, openai, local
DEEPSEEK_MODEL=deepseek-chat

# Trading Settings
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
MAX_POSITION_SIZE=0.1                 # 10% of balance
MAX_LEVERAGE=10

# Database (auto-configured with Docker)
DATABASE_URL=postgresql://cryptorl:cryptorl@localhost:5432/cryptorl
INFLUXDB_URL=http://localhost:8086
```

### Quick Test
```bash
# Test Binance connection
python scripts/test_api_connection.py

# Test LLM connection
python scripts/test_llm_connection.py

# Test data collection
python scripts/collect_data.py --mode test
```

## 📊 Dashboard Features

### Main Dashboard
- **Portfolio Overview**: Real-time P&L, positions, performance
- **Risk Management**: VaR calculations, drawdown monitoring
- **AI Signals**: LLM-generated trading recommendations
- **Performance Charts**: Historical performance, trade analysis

### Navigation
```
📊 Overview     - Key metrics & summary
💼 Portfolio   - Current positions & P&L
📈 Trading     - Live signals & execution
🛡️ Risk        - Risk metrics & alerts
📊 Performance - Historical analysis
📝 Logs        - System monitoring
```

## 🧪 Usage Examples

### 1. Basic Backtest
```python
# Run quick backtest
from cryptorl.backtesting.engine import BacktestingEngine

engine = BacktestingEngine()
results = engine.run_backtest(
    symbols=['BTCUSDT'],
    days=30,
    initial_balance=10000
)
print(f"Total Return: {results.total_return:.2f}%")
```

### 2. LLM Sentiment Analysis
```python
# Get market sentiment
from cryptorl.llm.confidence_generator import ConfidenceGenerator

generator = ConfidenceGenerator()
confidence = generator.generate_confidence('BTCUSDT', '2024-01-15')
print(f"Confidence: {confidence}")
```

### 3. Risk Monitoring
```python
# Check risk metrics
from cryptorl.risk_management.risk_manager import RiskManager

risk_manager = RiskManager()
metrics = risk_manager.calculate_portfolio_risk()
print(f"Portfolio VaR: ${metrics.var_95}")
```

## 🐳 Docker Quick Setup

### Minimal Setup
```bash
# Start databases only
docker-compose up -d influxdb postgresql redis

# Run dashboard
python run_dashboard.py
```

### Full Environment
```bash
# Complete stack
docker-compose up --build

# Access services:
# Dashboard: http://localhost:8501
# InfluxDB: http://localhost:8086 (admin/admin)
# PostgreSQL: localhost:5432 (cryptorl/cryptorl)
```

## 🔍 Troubleshooting

### Common Issues

**Dashboard crashes?**
```bash
# Use stable version
python run_dashboard.py

# Check logs
python quickstart.py --validate
```

**Import errors?**
```bash
# Fix Python path
export PYTHONPATH="${PYTHONPATH}:./src"

# Reinstall
uv pip install -e ".[dev]"
```

**API connection failed?**
```bash
# Test Binance
python scripts/test_binance_simple.py

# Check .env file
python quickstart.py --validate
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python quickstart.py --dashboard
```

## 🎯 Next Steps

### 1. Safe Testing (Recommended)
```bash
# Use testnet first
BINANCE_TESTNET=true python quickstart.py --dashboard

# Paper trading mode
python scripts/setup_testnet.py
```

### 2. Live Trading
```bash
# Switch to mainnet
# Edit .env: BINANCE_TESTNET=false
python quickstart.py --dashboard
```

### 3. Advanced Configuration
```bash
# Customize trading parameters
python scripts/setup_system.py --interactive
```

## 📈 Performance Check

### Quick Validation
```bash
# System health check
python quickstart.py --validate

# Run demo
python quick_start.py

# Check API connections
python scripts/test_api_connection.py
```

Expected successful output:
```
✅ System validation complete
✅ All services healthy
✅ Dashboard accessible at http://localhost:8501
✅ Ready for trading
```

## 🚨 Safety Notes

- **Always use testnet first**
- **Start with small amounts**
- **Monitor risk closely**
- **Test backtests before live trading**

---

**Ready to start?** Run `python quickstart.py --setup` and begin your AI trading journey!