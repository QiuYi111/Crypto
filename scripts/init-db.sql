-- Initialize CryptoRL PostgreSQL database
-- This script runs automatically when the PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schema
CREATE SCHEMA IF NOT EXISTS cryptorl;
SET search_path TO cryptorl, public;

-- Create tables
CREATE TABLE IF NOT EXISTS trading_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_name VARCHAR(255) NOT NULL UNIQUE,
    exchange VARCHAR(50) NOT NULL,
    api_key_hash VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trading_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES trading_accounts(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    leverage DECIMAL(5, 2) DEFAULT 1.0,
    margin DECIMAL(20, 8) NOT NULL,
    liquidation_price DECIMAL(20, 8),
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'LIQUIDATED')),
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trading_orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES trading_accounts(id),
    symbol VARCHAR(20) NOT NULL,
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'TAKE_PROFIT')),
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'FILLED', 'CANCELLED', 'REJECTED')),
    binance_order_id BIGINT,
    client_order_id VARCHAR(255),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    filled_price DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market_data_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(20, 8) NOT NULL,
    high_price DECIMAL(20, 8) NOT NULL,
    low_price DECIMAL(20, 8) NOT NULL,
    close_price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_volume DECIMAL(20, 8),
    trades_count INTEGER,
    funding_rate DECIMAL(10, 8),
    next_funding_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS confidence_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    fundamentals_score DECIMAL(3, 2) CHECK (fundamentals_score BETWEEN 0 AND 1),
    industry_score DECIMAL(3, 2) CHECK (industry_score BETWEEN 0 AND 1),
    geopolitics_score DECIMAL(3, 2) CHECK (geopolitics_score BETWEEN 0 AND 1),
    macroeconomics_score DECIMAL(3, 2) CHECK (macroeconomics_score BETWEEN 0 AND 1),
    news_sources TEXT[],
    llm_model VARCHAR(100),
    confidence_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, date)
);

CREATE TABLE IF NOT EXISTS rl_training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    model_config JSONB,
    training_config JSONB,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    total_episodes INTEGER,
    final_reward FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    win_rate FLOAT,
    model_path VARCHAR(500),
    status VARCHAR(20) DEFAULT 'RUNNING' CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES rl_training_runs(id),
    symbol VARCHAR(20) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_balance DECIMAL(20, 8) NOT NULL,
    final_balance DECIMAL(20, 8) NOT NULL,
    total_return DECIMAL(10, 4) NOT NULL,
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(5, 2),
    profit_factor DECIMAL(10, 4),
    total_trades INTEGER,
    avg_trade_return DECIMAL(10, 4),
    backtest_config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trading_positions_account_id ON trading_positions(account_id);
CREATE INDEX IF NOT EXISTS idx_trading_positions_symbol ON trading_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_positions_status ON trading_positions(status);
CREATE INDEX IF NOT EXISTS idx_trading_positions_opened_at ON trading_positions(opened_at);

CREATE INDEX IF NOT EXISTS idx_trading_orders_account_id ON trading_orders(account_id);
CREATE INDEX IF NOT EXISTS idx_trading_orders_symbol ON trading_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_orders_status ON trading_orders(status);
CREATE INDEX IF NOT EXISTS idx_trading_orders_created_at ON trading_orders(created_at);

CREATE INDEX IF NOT EXISTS idx_market_data_snapshots_symbol_timestamp ON market_data_snapshots(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_snapshots_timestamp ON market_data_snapshots(timestamp);

CREATE INDEX IF NOT EXISTS idx_confidence_vectors_symbol_date ON confidence_vectors(symbol, date);
CREATE INDEX IF NOT EXISTS idx_confidence_vectors_date ON confidence_vectors(date);

CREATE INDEX IF NOT EXISTS idx_rl_training_runs_algorithm ON rl_training_runs(algorithm);
CREATE INDEX IF NOT EXISTS idx_rl_training_runs_status ON rl_training_runs(status);
CREATE INDEX IF NOT EXISTS idx_rl_training_runs_created_at ON rl_training_runs(created_at);

CREATE INDEX IF NOT EXISTS idx_backtest_results_run_id ON backtest_results(run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol ON backtest_results(symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_results_start_date_end_date ON backtest_results(start_date, end_date);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON trading_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_positions_updated_at BEFORE UPDATE ON trading_positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_orders_updated_at BEFORE UPDATE ON trading_orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_confidence_vectors_updated_at BEFORE UPDATE ON confidence_vectors
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();