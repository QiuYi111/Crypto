#!/bin/bash

# CryptoRL Launch Script
# Easy-to-use wrapper for common CryptoRL operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
ENV_FILE=".env"
COMPOSE_FILE="docker-compose.yml"

# Help function
show_help() {
    echo -e "${BLUE}CryptoRL Launch Script${NC}"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup        - Initial setup and configuration"
    echo "  start        - Start databases and core services"
    echo "  stop         - Stop all services"
    echo "  restart      - Restart all services"
    echo "  status       - Check service status"
    echo "  collect      - Collect historical market data"
    echo "  train        - Start RL training"
    echo "  backtest     - Run backtesting"
    echo "  dashboard    - Start Streamlit dashboard"
    echo "  jupyter      - Start Jupyter notebook server"
    echo "  logs         - Show logs for services"
    echo "  clean        - Clean up containers and volumes"
    echo "  test         - Run tests"
    echo "  help         - Show this help message"
    echo ""
    echo "Options:"
    echo "  -e, --env FILE     Use custom environment file (default: .env)"
    echo "  -h, --help         Show help"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # Initial setup"
    echo "  $0 start                   # Start databases"
    echo "  $0 collect --symbol BTCUSDT # Collect BTC data"
    echo "  $0 train --model ppo        # Start PPO training"
    echo "  $0 dashboard               # Launch dashboard"
}

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed. Please install Python3 first."
        exit 1
    fi
    
    if [[ -f "$SCRIPT_DIR/uv.lock" ]] && ! command -v uv &> /dev/null; then
        log_warn "uv is not installed but uv.lock found. Install uv for better performance."
    fi
    
    log_info "Prerequisites check passed"
}

# Setup function
setup_system() {
    log_info "Setting up CryptoRL system..."
    
    check_prerequisites
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$SCRIPT_DIR/$ENV_FILE" ]]; then
        log_warn ".env file not found, creating from template..."
        cat > "$SCRIPT_DIR/$ENV_FILE" << EOF
# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=true

# Database Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=cryptorl_token_2024
INFLUXDB_ORG=cryptorl
INFLUXDB_BUCKET=market_data

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=cryptorl
POSTGRES_USER=cryptorl
POSTGRES_PASSWORD=cryptorl

# Redis Configuration
REDIS_URL=redis://localhost:6379

# LLM Configuration
LLM_MODEL_PATH=./models
LLM_API_URL=http://localhost:8000

# Training Configuration
TRAINING_DATA_DAYS=30
BATCH_SIZE=32
LEARNING_RATE=0.001
EOF
        log_info "Created .env file. Please edit it with your actual API keys."
    fi
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    cd "$SCRIPT_DIR"
    
    if command -v uv &> /dev/null; then
        log_info "Using uv for package management..."
        uv pip install -e .
        uv pip install -e ".[dev]"
    else
        log_info "Using pip for package management..."
        pip install -e .
        pip install -e ".[dev]"
    fi
    
    log_info "Setup complete!"
}

# Start services
start_services() {
    log_info "Starting CryptoRL services..."
    
    cd "$SCRIPT_DIR"
    
    # Start databases
    docker compose up -d influxdb postgresql redis
    
    # Wait for services to be ready
    log_info "Waiting for services to initialize..."
    sleep 10
    
    # Check service health
    check_service_health
    
    log_info "Services started successfully!"
}

# Stop services
stop_services() {
    log_info "Stopping CryptoRL services..."
    
    cd "$SCRIPT_DIR"
    docker compose down
    
    log_info "Services stopped."
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    # Check InfluxDB
    if curl -s http://localhost:8086/health > /dev/null; then
        log_info "✓ InfluxDB is running"
    else
        log_error "✗ InfluxDB is not responding"
    fi
    
    # Check PostgreSQL
    if docker exec cryptorl-postgresql pg_isready -U cryptorl -d cryptorl > /dev/null 2>&1; then
        log_info "✓ PostgreSQL is running"
    else
        log_error "✗ PostgreSQL is not responding"
    fi
    
    # Check Redis
    if docker exec cryptorl-redis redis-cli ping > /dev/null 2>&1; then
        log_info "✓ Redis is running"
    else
        log_error "✗ Redis is not responding"
    fi
}

# Collect data
collect_data() {
    log_info "Starting data collection..."
    
    local symbol="${1:-BTCUSDT}"
    local days="${2:-30}"
    
    cd "$SCRIPT_DIR"
    
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
    
    python "$SCRIPT_DIR/scripts/init_influxdb.py" && python "$SCRIPT_DIR/scripts/collect_data.py" --symbols "$symbol" --days "$days"
    
    log_info "Data collection completed for $symbol"
}

# Start training
start_training() {
    log_info "Starting RL training..."
    
    local algorithm="${1:-ppo}"
    local episodes="${2:-1000}"
    
    cd "$SCRIPT_DIR"
    
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
    
    python "$SCRIPT_DIR/scripts/phase4_demo.py" --algorithm "$algorithm" --episodes "$episodes"
    
    log_info "Training session completed"
}

# Run backtesting
run_backtest() {
    log_info "Running backtesting..."
    
    local strategy="${1:-ppo}"
    local days="${2:-30}"
    
    cd "$SCRIPT_DIR"
    
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
    
    python "$SCRIPT_DIR/scripts/phase4_demo.py" --mode backtest --strategy "$strategy" --days "$days"
    
    log_info "Backtesting completed"
}

# Start dashboard
start_dashboard() {
    log_info "Starting Streamlit dashboard..."
    
    cd "$SCRIPT_DIR"
    
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
    
    streamlit run "$SCRIPT_DIR/src/cryptorl/monitoring/dashboard.py" --server.port 8501
}

# Start Jupyter
start_jupyter() {
    log_info "Starting Jupyter notebook server..."
    
    cd "$SCRIPT_DIR"
    
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
    
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='cryptorl'
}

# Show logs
show_logs() {
    local service="${1:-all}"
    
    cd "$SCRIPT_DIR"
    
    if [[ "$service" == "all" ]]; then
        docker compose logs -f
    else
        docker compose logs -f "$service"
    fi
}

# Clean up
clean_up() {
    log_warn "Cleaning up containers and volumes..."
    
    cd "$SCRIPT_DIR"
    docker compose down -v --remove-orphans
    docker system prune -f
    
    log_info "Cleanup completed"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd "$SCRIPT_DIR"
    
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
    
    python -m pytest tests/ -v
    
    log_info "Tests completed"
}

# Main function
main() {
    case "${1:-help}" in
        setup)
            setup_system
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            start_services
            ;;
        status)
            check_service_health
            ;;
        collect)
            collect_data "${2:-BTCUSDT}" "${3:-1h}" "${4:-30}"
            ;;
        train)
            start_training "${2:-ppo}" "${3:-1000}"
            ;;
        backtest)
            run_backtest "${1:-ppo}" "${2:-30}"
            ;;
        dashboard)
            start_dashboard
            ;;
        jupyter)
            start_jupyter
            ;;
        logs)
            show_logs "${2:-all}"
            ;;
        clean)
            clean_up
            ;;
        test)
            run_tests
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

main "$@"