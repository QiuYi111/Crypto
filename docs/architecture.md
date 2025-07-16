# CryptoRL Architecture Guide

This document provides a detailed overview of the CryptoRL system architecture, showing how all components work together.

## System Overview

CryptoRL is a production-ready crypto trading RL system with 5 completed phases. The architecture has been simplified for deployment while maintaining modularity.

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface                               │
│                 (Streamlit Dashboard)                           │
├─────────────────────────────────────────────────────────────────┤
│                   Core Services                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Data      │  │    LLM      │  │   Trading   │             │
│  │ Collection  │  │  Analysis   │  │   Engine    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                  Storage Layer                                  │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │  InfluxDB   │  │ PostgreSQL  │                               │
│  │ Time Series │  │ Relational  │                               │
│  └─────────────┘  └─────────────┘                               │
├─────────────────────────────────────────────────────────────────┤
│                  Quick Setup                                    │
│                 (single script)                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Collection (`src/cryptorl/data/`)

Responsible for all market data acquisition and storage.

#### Key Classes:

- **BinanceClient** (`binance_client.py`)
  - Handles all Binance API interactions
  - Rate limiting and retry logic
  - WebSocket support for real-time data
  ```python
  from cryptorl.data import BinanceClient
  
  client = BinanceClient()
  klines = client.get_klines("BTCUSDT", "4h", limit=1000)
  ```

- **MarketData** (`market_data.py`)
  - Historical data collection orchestration
  - Data validation and cleaning
  - Multi-threaded collection
  ```python
  from cryptorl.data import MarketData
  
  md = MarketData()
  md.collect_historical(
      symbols=["BTCUSDT", "ETHUSDT"],
      interval="1h",
      days=30
  )
  ```

- **InfluxDBClient** (`influxdb_client.py`)
  - Time-series data storage
  - Optimized for financial data queries
  - Automatic retention policies

### 2. LLM Integration (`src/cryptorl/llm/`)

Provides sentiment analysis and confidence scoring using LLMs.

#### Key Classes:

- **LLMClient** (`llm_client.py`)
  - Unified interface for different LLM providers
  - Local and cloud model support
  - Caching for performance

- **RAGPipeline** (`rag_pipeline.py`)
  - Retrieval-Augmented Generation
  - News search and context building
  - Confidence vector generation
  ```python
  from cryptorl.llm import RAGPipeline
  
  rag = RAGPipeline()
  confidence = rag.generate_confidence(
      symbol="BTCUSDT",
      date="2024-01-15",
      news_context=True
  )
  # Returns: [0.7, 0.6, 0.4, 0.8]  # [fundamentals, industry, geopolitics, macro]
  ```

- **ConfidenceGenerator** (`confidence_generator.py`)
  - 7-dimensional confidence vectors
  - Market sentiment analysis
  - Risk assessment integration

### 3. Reinforcement Learning (`src/cryptorl/rl/`)

Core RL training and agent management.

#### Key Classes:

- **TradingEnvironment** (`environment.py`)
  - Custom Gym environment for crypto trading
  - Multi-asset support
  - Realistic transaction costs
  ```python
  from cryptorl.rl import TradingEnvironment
  
  env = TradingEnvironment(
      symbols=["BTCUSDT", "ETHUSDT"],
      initial_balance=10000,
      confidence_data=True
  )
  ```

- **MambaExploration** (`mamba_exploration.py`)
  - Experimental Mamba architecture implementation
  - State space modeling for time series
  - Performance benchmarking

- **TrainingPipeline** (`training.py`)
  - Multi-algorithm support (PPO, SAC, TD3)
  - Distributed training capabilities
  - Model versioning and tracking

### 4. Trading System (`src/cryptorl/trading/`)

Live trading execution and order management.

#### Key Classes:

- **ExecutionEngine** (`execution.py`)
  - Order placement and management
  - Risk controls and position sizing
  - Paper trading and live modes

- **RiskManager** (`risk_manager.py`)
  - Dynamic position sizing
  - Stop-loss and take-profit
  - Portfolio-level risk metrics

### 5. Monitoring (`src/cryptorl/monitoring/`)

Real-time monitoring and alerting.

#### Key Classes:

- **Dashboard** (`dashboard.py`)
  - Streamlit-based web dashboard
  - Real-time PnL tracking
  - Model performance metrics

## Data Flow

### 1. Historical Data Collection

```
Binance API → MarketData → InfluxDB → RL Environment
     ↓
News APIs → RAG Pipeline → PostgreSQL → Confidence Vectors
```

### 2. Live Trading

```
WebSocket → Real-time Data → RL Agent → Trading Engine → Binance
     ↓           ↓              ↓           ↓
InfluxDB → Dashboard → Risk Manager → Execution
```

### 3. Training Pipeline

```
Historical Data → Data Fusion → RL Environment → Agent Training → Model Registry
     ↓              ↓              ↓               ↓              ↓
InfluxDB    PostgreSQL    Gym Environment    Stable-Baselines3    Models/
```

## Database Schema

### InfluxDB (Time Series)

```
measurements:
  - ohlcv_data
    tags: symbol, interval
    fields: open, high, low, close, volume
  
  - funding_rates
    tags: symbol
    fields: funding_rate, funding_time
  
  - order_book
    tags: symbol
    fields: bids, asks, timestamp
```

### PostgreSQL (Relational)

```sql
-- Core tables
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    side VARCHAR(10),
    quantity DECIMAL,
    price DECIMAL,
    timestamp TIMESTAMP
);

CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    side VARCHAR(10),
    quantity DECIMAL,
    entry_price DECIMAL,
    current_price DECIMAL,
    unrealized_pnl DECIMAL
);

CREATE TABLE confidence_vectors (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    date DATE,
    fundamentals FLOAT,
    industry FLOAT,
    geopolitics FLOAT,
    macro FLOAT,
    technical FLOAT,
    sentiment FLOAT,
    risk FLOAT
);
```

## Configuration Management

### Environment Variables (Actual)

Current configuration uses Pydantic v2 with comprehensive settings:

```bash
# Required: Binance API
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
BINANCE_TESTNET=true

# Required: Database
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influx_token
POSTGRESQL_URL=postgresql://cryptorl:cryptorl@localhost:5432/cryptorl

# LLM Provider Selection:
LLM_PROVIDER=deepseek|openai|local
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key  # if using OpenAI

# Search APIs:
SERPAPI_KEY=your_serpapi_key
GOOGLE_API_KEY=your_google_key
GOOGLE_CX=your_search_cx

# Optional: China-compatible
BAIDU_API_KEY=your_baidu_key

# RL Settings:
RL_USE_MAMBA=true
RL_INITIAL_BALANCE=10000.0
RL_MAX_POSITION_SIZE=1.0
```

### Actual Configuration Structure

```python
# src/cryptorl/config/settings.py - Pydantic v2
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Binance
    binance_api_key: str
    binance_secret_key: str
    binance_testnet: bool = True
    
    # Database
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str
    postgresql_url: str = "postgresql://cryptorl:cryptorl@localhost:5432/cryptorl"
    
    # LLM Provider
    llm_provider: str = "local"  # deepseek, openai, local
    deepseek_api_key: str = ""
    openai_api_key: str = ""
    
    # Trading
    trading_symbols: list = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    max_position_size: float = 0.1
    max_leverage: int = 10
    
    # RL
    rl_use_mamba: bool = True
    rl_initial_balance: float = 10000.0
    rl_learning_rate: float = 3e-4
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## Development Workflow

### 1. Quick Start (Current)

```bash
# Single command setup
python quickstart.py --setup

# Validate configuration
python quickstart.py --validate

# Start dashboard
python quickstart.py --dashboard

# Test system
python quickstart.py --test
```

### 2. Manual Setup

```bash
# Install uv if needed
python quickstart.py --setup

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run validation
python scripts/validate_setup.py
```

### 3. Testing (Built-in)

```bash
# All tests via quickstart
python quickstart.py --test

# Manual validation
python scripts/validate_setup.py

# API connection test
python scripts/test_api_connection.py
```

### 4. Model Training (Phase 3 Complete)

```bash
# Use phase3 demo scripts
python scripts/phase3_demo.py    # Mamba vs Transformer benchmarks
python scripts/phase4_demo.py    # Full backtesting system

# Training via train.py
python train.py --config config/training_config.json
```

## Deployment Patterns (Current)

### 1. Local Development (Working)

```bash
# Single script setup
python quickstart.py --setup

# Docker services (if needed)
docker compose up -d

# Validation
python scripts/validate_setup.py
```

### 2. Production Setup

```bash
# Environment setup
cp .env.example .env
# Edit .env with production values

# Validate configuration
python quickstart.py --validate

# Start dashboard
python quickstart.py --dashboard
```

### 3. Current Architecture

- **Database**: InfluxDB + PostgreSQL via Docker
- **LLM**: DeepSeek API (default) or local models
- **RL Training**: Mamba-based with Stable-Baselines3
- **Monitoring**: Streamlit dashboard
- **Validation**: Built-in validation scripts

## Performance Considerations

### 1. Data Storage

- **InfluxDB**: Optimized for time-series queries
- **Partitioning**: Data partitioned by day and symbol
- **Retention**: Automatic cleanup of old data

### 2. Model Serving

- **Model caching**: Models loaded into memory
- **Batch processing**: Efficient batch inference
- **GPU utilization**: CUDA acceleration for LLM and RL

### 3. Network Optimization

- **Connection pooling**: Database connection reuse
- **API rate limiting**: Respecting exchange limits
- **WebSocket multiplexing**: Efficient real-time data

## Security Considerations

### 1. API Security

- **Key rotation**: Regular API key updates
- **IP whitelisting**: Restrict API access
- **Secret management**: Environment variables and vaults

### 2. Data Security

- **Encryption at rest**: Database encryption
- **Encryption in transit**: TLS for all connections
- **Access control**: Role-based permissions

### 3. Trading Safety

- **Testnet first**: Always test in sandbox
- **Circuit breakers**: Automatic trading halts
- **Risk limits**: Hard-coded position limits

## Current Status & Next Steps

### ✅ Actually Completed Components
- **Phase 1**: ✅ Data Collection & Storage (InfluxDB + PostgreSQL)
- **Phase 2**: ✅ LLM Integration & 7D Confidence Vectors
- **Phase 3**: ✅ Mamba vs Transformer Benchmarking (results in reports/)
- **Phase 4**: ✅ Backtesting & Risk Management (reports/phase4_summary.json)
- **Phase 5**: ✅ Dashboard & Monitoring (simple_dashboard.py + run_dashboard.py)

### 🔄 Actual Current State
- **Quickstart**: Unified `quickstart.py` replaces all test scripts
- **Validation**: `scripts/validate_setup.py` tests entire system
- **Dashboard**: Two versions - `simple_dashboard.py` (stable) and `run_dashboard.py`
- **Dependencies**: Pyproject.toml with uv/conda support
- **Configuration**: Pydantic v2 settings with provider selection (DeepSeek/OpenAI/Local)

### 📋 Actual Usage
```bash
# Setup
python quickstart.py --setup

# Configure
cp .env.example .env  # then edit with your keys

# Validate
python quickstart.py --validate

# Dashboard
python quickstart.py --dashboard

# Full validation
python scripts/validate_setup.py
```

### 🎯 Next Steps (Real)
1. Configure `.env` with actual API keys
2. Run validation to test all components
3. Start dashboard for monitoring
4. Run `scripts/phase3_demo.py` for Mamba benchmarks
5. Run `scripts/phase4_demo.py` for backtesting demo