# CryptoRL Architecture Guide

This document provides a detailed overview of the CryptoRL system architecture, showing how all components work together.

## System Overview

CryptoRL is built with a modular architecture that separates concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    API Layer                                    │
├─────────────────────────────────────────────────────────────────┤
│                   Service Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Data      │  │    LLM      │  │   Trading   │             │
│  │ Collection  │  │  Analysis   │  │   Engine    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                  Storage Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  InfluxDB   │  │ PostgreSQL  │  │    Redis    │             │
│  │ Time Series │  │ Relational  │  │    Cache    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                  Infrastructure                                 │
│                    Docker/K8s                                   │
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

### Environment Variables

All configuration is managed through environment variables and `.env` files:

```bash
# Core settings
CRYPTORL_ENV=development
CRYPTORL_LOG_LEVEL=INFO

# Binance
BINANCE_API_KEY=xxx
BINANCE_SECRET_KEY=xxx
BINANCE_TESTNET=true

# Database
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=xxx
POSTGRESQL_URL=postgresql://user:pass@localhost:5432/cryptorl

# LLM
LLM_MODEL_PATH=/models/llama-2-7b-chat
LLM_DEVICE=cuda
LLM_BATCH_SIZE=8
```

### Configuration Files

```python
# src/cryptorl/config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Binance settings
    binance_api_key: str
    binance_secret_key: str
    binance_testnet: bool = True
    
    # Database settings
    influxdb_url: str
    influxdb_token: str
    postgresql_url: str
    
    # LLM settings
    llm_model_path: str
    llm_device: str = "cuda"
    llm_max_tokens: int = 512
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## Development Workflow

### 1. Local Development

```bash
# Start development environment
docker compose --profile dev up -d

# Install dependencies
poetry install

# Run tests
pytest tests/

# Start dashboard
streamlit run src/cryptorl/monitoring/dashboard.py
```

### 2. Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# Load tests
pytest tests/load/
```

### 3. Model Training

```bash
# Train baseline model
python scripts/train_model.py --algorithm PPO --symbols BTCUSDT ETHUSDT --days 30

# Evaluate model
python scripts/evaluate_model.py --model-path models/ppo_btc_eth_v1

# Compare algorithms
python scripts/compare_algorithms.py --symbols BTCUSDT --days 60
```

## Deployment Patterns

### 1. Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'
services:
  cryptorl:
    build: .
    environment:
      - CRYPTORL_ENV=development
    depends_on:
      - influxdb
      - postgresql
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

### 2. Kubernetes (Production)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cryptorl-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cryptorl-agent
  template:
    spec:
      containers:
      - name: cryptorl
        image: cryptorl:latest
        env:
        - name: CRYPTORL_ENV
          value: "production"
```

### 3. Cloud Deployment

- **AWS**: ECS with RDS PostgreSQL and Timestream
- **GCP**: Cloud Run with Cloud SQL and BigQuery
- **Azure**: Container Apps with PostgreSQL and Time Series Insights

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