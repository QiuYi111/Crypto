# CryptoRL Agent

A comprehensive cryptocurrency day trading reinforcement learning agent with LLM-enhanced sentiment analysis.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Binance API credentials
- InfluxDB token (auto-generated in Docker setup)

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd cryptorl-agent

# Copy environment configuration
cp .env.example .env

# Edit .env with your API keys and configuration
nano .env
```

### 2. Start Infrastructure

```bash
# Start databases and services
docker compose up -d influxdb postgresql redis

# Verify services are running
docker compose ps
```

### 3. Install Dependencies

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 4. Collect Historical Data

```bash
# Collect 30 days of historical data for BTC, ETH, SOL
python scripts/collect_data.py --mode historical --days 30

# Or collect specific symbols
python scripts/collect_data.py --symbols BTCUSDT ETHUSDT --days 7
```

## 📁 Project Structure

```
cryptorl-agent/
├── src/cryptorl/
│   ├── data/           # Market data collection and storage
│   │   ├── binance_client.py    # Binance API wrapper
│   │   ├── market_data.py       # Historical data collection
│   │   └── influxdb_client.py   # InfluxDB time-series storage
│   ├── llm/            # LLM integration for sentiment analysis
│   ├── rl/             # Reinforcement learning environment and agents
│   ├── trading/        # Trading execution and risk management
│   ├── monitoring/     # Real-time dashboard and monitoring
│   └── config/         # Configuration management
├── scripts/
│   ├── collect_data.py # Data collection script
│   └── init-db.sql     # Database initialization
├── docker/
│   ├── Dockerfile      # Main application container
│   └── docker-compose.yml # Development environment
├── tests/              # Unit and integration tests
├── models/             # Trained RL models
├── data/               # Local data storage
└── logs/               # Application logs
```

## 🔧 Configuration

### Environment Variables

Key configuration in `.env`:

```bash
# Binance API
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
BINANCE_TESTNET=true  # Use testnet for development

# Database
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influx_token
POSTGRESQL_URL=postgresql://cryptorl:cryptorl@localhost:5432/cryptorl

# LLM Configuration
LLM_MODEL_PATH=/models/llama-2-7b-chat
LLM_DEVICE=cuda
```

### Database Access

```bash
# PostgreSQL
psql postgresql://cryptorl:cryptorl@localhost:5432/cryptorl

# InfluxDB
# Visit http://localhost:8086 (admin/cryptorl_admin_2024)
```

## 📊 Data Collection

### Supported Intervals
- 1m, 3m, 5m, 15m, 30m
- 1h, 2h, 4h, 6h, 8h, 12h
- 1d, 3d, 1w, 1M

### Symbol Support
- BTCUSDT, ETHUSDT, SOLUSDT (default)
- Any Binance Futures trading pair

### Data Storage
- **Time-series data**: InfluxDB (OHLCV, funding rates, real-time snapshots)
- **Relational data**: PostgreSQL (accounts, positions, orders, backtests)

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m slow
```

### Development Environment

```bash
# Start full development environment
docker compose --profile dev up -d

# Access Jupyter notebook
# Visit http://localhost:8888 (token: cryptorl)

# Access Streamlit dashboard
# Visit http://localhost:8501
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

## 🎯 Next Steps

1. **Phase 2**: LLM Integration
   - Set up local LLM service
   - Implement news sentiment analysis
   - Generate confidence vectors

2. **Phase 3**: RL Training
   - Create training environment
   - Implement Mamba architecture
   - Train baseline models (PPO, SAC)

3. **Phase 4**: Trading System
   - Implement risk management
   - Set up paper trading
   - Build backtesting framework

4. **Phase 5**: Production
   - Deploy monitoring dashboard
   - Set up alerts and logging
   - Run live trading

## 📈 Monitoring

### Grafana Dashboard
Access Grafana at `http://localhost:3000` (admin/admin) to monitor:
- Database performance
- API rate limits
- Data collection metrics
- System health

### Logs
```bash
# View application logs
tail -f logs/cryptorl.log

# View container logs
docker compose logs -f cryptorl
```

## 🛡️ Security

- All secrets managed via environment variables
- Testnet enabled by default for safe development
- Rate limiting on all API calls
- Input validation and sanitization

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and add tests
3. Run tests: `pytest`
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- Check logs: `docker compose logs`
- Database issues: Ensure services are running with `docker compose ps`
- API issues: Verify Binance API keys in `.env`
- Performance: Check resource usage with `docker stats`