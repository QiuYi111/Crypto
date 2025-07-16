# 🚀 CryptoRL Agent Quick Start

Get your CryptoRL trading agent up and running in minutes!

## ⚡ One-Command Setup

```bash
python quickstart.py --setup
```

This will:
- ✅ Check Python version (3.8+)
- ✅ Install `uv` package manager (if needed)
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Generate `.env` template
- ✅ Set up configuration files

## 🎯 Next Steps

1. **Configure API credentials**:
   ```bash
   # Edit .env file with your credentials
   nano .env
   ```

2. **Validate configuration**:
   ```bash
   python quickstart.py --validate
   ```

3. **Run tests**:
   ```bash
   python quickstart.py --test
   ```

4. **Start dashboard**:
   ```bash
   python quickstart.py --dashboard
   ```

## 📋 Quick Commands

| Command | Description |
|---------|-------------|
| `python quickstart.py --setup` | Complete environment setup + start Docker services |
| `python quickstart.py --validate` | Check configuration |
| `python quickstart.py --test` | Run validation tests |
| `python quickstart.py --dashboard` | Start monitoring dashboard |
| `python quickstart.py --docker-up` | Start Docker services (databases) |
| `python quickstart.py --docker-down` | Stop Docker services |
| `python quickstart.py --help` | Show help |

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

### Using UV (Recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"
```

### Using pip
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ".[dev]"
```

## 🔑 Required Configuration

Update `.env` file with your credentials:

```bash
BINANCE_API_KEY=your_actual_api_key
BINANCE_SECRET_KEY=your_actual_secret_key
DEEPSEEK_API_KEY=your_deepseek_api_key
DATABASE_URL=postgresql://cryptorl:cryptorl@postgresql:5432/cryptorl
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=cryptorl_token_2024
REDIS_URL=redis://redis:6379/0
```

## 🐳 Docker Setup

```bash
# Build and run
docker-compose up --build

# Or run individual services
docker build -t cryptorl-agent .
docker run -it --env-file .env cryptorl-agent
```

## 📊 Access Points

- **Dashboard**: http://localhost:8501
- **InfluxDB**: http://localhost:8086 (admin/cryptorl_admin_2024)
- **PostgreSQL**: localhost:5432 (cryptorl/cryptorl)
- **Redis**: localhost:6379
- **Logs**: `logs/cryptorl.log`

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Run `python quickstart.py --test`
2. **Configuration issues**: Run `python quickstart.py --validate`
3. **Missing dependencies**: Run `python quickstart.py --setup`
4. **Permission errors**: Check file permissions with `ls -la`

### Debug Mode

```bash
export LOG_LEVEL=DEBUG
python quickstart.py --setup
```

### Reset Environment

```bash
rm -rf .venv
python quickstart.py --setup
```

## 📞 Support

- **Logs**: Check `logs/cryptorl.log` for detailed errors
- **Issues**: Report bugs via GitHub issues
- **Config**: Verify `.env` file contains valid credentials
- **Services**: Ensure Docker services are running (if using Docker)