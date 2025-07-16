# Getting Started with CryptoRL

This guide will walk you through setting up and using the CryptoRL trading agent from scratch.

## Overview

CryptoRL is an advanced cryptocurrency day trading system that combines:
- **Reinforcement Learning** for trading decisions
- **Large Language Models** for sentiment analysis and market insights
- **Multi-market training** across different crypto assets
- **Risk management** and portfolio optimization

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Processing      │    │   Trading       │
│                 │    │                  │    │                 │
│ • Binance API   │───▶│ • LLM Analysis   │───▶│ • RL Agent      │
│ • News Sources  │    │ • Data Fusion    │    │ • Risk Mgmt     │
│ • Time Series   │    │ • Confidence     │    │ • Execution     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Prerequisites

Before starting, ensure you have:
- Python 3.9+ installed
- Docker and Docker Compose
- Binance API credentials (testnet recommended for development)
- Minimum 8GB RAM (16GB+ recommended for LLM processing)
- 50GB free disk space for historical data

## Quick Setup

### 1. Environment Setup

```bash
# Clone and navigate
git clone <repository-url>
cd cryptorl-agent

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env  # Add your API keys
```

### 2. Infrastructure Launch

```bash
# Start databases and services
docker compose up -d influxdb postgresql redis

# Verify everything is running
docker compose ps
```

Expected output:
```
NAME                  STATUS          PORTS
influxdb             healthy         0.0.0.0:8086->8086/tcp
postgresql           healthy         0.0.0.0:5432->5432/tcp
redis                healthy         0.0.0.0:6379->6379/tcp
```

### 3. Install Dependencies

```bash
# Using Poetry (recommended)
pip install poetry
poetry install
poetry shell

# Or using pip
pip install -e .
```

### 4. Initial Data Collection

```bash
# Collect 7 days of data for BTC and ETH
python scripts/collect_data.py --symbols BTCUSDT ETHUSDT --days 7

# Monitor progress
tail -f logs/collect_data.log
```

### 5. Verify Setup

```bash
# Test all connections
python scripts/validate_setup.py

# Check system status
python scripts/check_config.py
```

## Your First Trading Session

### 1. Start the Dashboard

```bash
# Launch monitoring dashboard
streamlit run src/cryptorl/monitoring/dashboard.py

# Access at http://localhost:8501
```

### 2. Run Paper Trading

```bash
# Start paper trading with testnet
python scripts/live_test.py --mode paper --symbols BTCUSDT

# View real-time logs
tail -f logs/live_trading.log
```

### 3. Monitor Performance

- **Dashboard**: http://localhost:8501
- **Grafana**: http://localhost:3000 (admin/admin)
- **InfluxDB**: http://localhost:8086 (admin/cryptorl_admin_2024)

## Common First Steps

### Adding New Symbols

```python
# In Python
from cryptorl.data import BinanceClient

client = BinanceClient()
client.collect_historical_data(
    symbols=["ADAUSDT", "DOTUSDT"],
    interval="4h",
    days=30
)
```

### Custom LLM Configuration

```python
# Configure LLM settings
from cryptorl.llm import LLMClient

llm = LLMClient(
    model_path="/models/custom-llama",
    device="cuda",
    max_tokens=512
)
```

### Basic RL Training

```python
# Train a simple PPO agent
from cryptorl.rl import TrainingPipeline

pipeline = TrainingPipeline(
    algorithm="PPO",
    symbols=["BTCUSDT", "ETHUSDT"],
    days=30
)
pipeline.train(episodes=1000)
```

## Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check if services are running
docker compose ps

# Restart services
docker compose restart influxdb postgresql
```

**Binance API Errors**
```bash
# Test API connection
python scripts/test_api_connection.py

# Verify credentials in .env
echo $BINANCE_API_KEY
```

**Memory Issues**
```bash
# Check memory usage
docker stats

# Reduce LLM batch size in config
export LLM_BATCH_SIZE=4
```

### Getting Help

1. **Check logs**: `docker compose logs -f`
2. **Database issues**: Ensure services are healthy
3. **API issues**: Verify credentials and rate limits
4. **Performance**: Monitor resource usage

## Next Steps

1. **Explore the codebase** with the [Architecture Guide](architecture.md)
2. **Configure LLM** following [LLM Setup](llm-setup.md)
3. **Set up training** with [Training Guide](training.md)
4. **Deploy for production** using [Deployment Guide](deployment.md)