# Reinforcement Learning Training Guide

This guide covers training reinforcement learning agents for cryptocurrency trading using CryptoRL.

## Overview

CryptoRL supports multiple RL algorithms and architectures:
- **PPO** (Proximal Policy Optimization) - Stable and reliable
- **SAC** (Soft Actor-Critic) - Good for continuous actions
- **TD3** (Twin Delayed DDPG) - Handles overestimation bias
- **Mamba** - Experimental state-space model architecture

## Training Pipeline

### 1. Environment Setup

#### Basic Environment Configuration
```python
from cryptorl.rl import TradingEnvironment

# Create environment for single symbol
env = TradingEnvironment(
    symbols=["BTCUSDT"],
    interval="4h",
    initial_balance=10000,
    confidence_data=True,
    transaction_cost=0.001,  # 0.1% trading fee
    max_positions=1,
    slippage=0.0005
)

# Multi-symbol environment
multi_env = TradingEnvironment(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    interval="1h",
    initial_balance=10000,
    confidence_data=True,
    max_positions=3
)
```

#### Advanced Environment Settings
```python
# Risk management parameters
env = TradingEnvironment(
    symbols=["BTCUSDT"],
    risk_params={
        "max_drawdown": 0.1,      # 10% max drawdown
        "max_position_size": 0.5, # 50% max position
        "stop_loss": 0.05,        # 5% stop loss
        "take_profit": 0.1,       # 10% take profit
    },
    reward_params={
        "sharpe_weight": 1.0,
        "drawdown_weight": 2.0,
        "profit_weight": 1.0
    }
)
```

### 2. Data Preparation

#### Historical Data Collection
```python
from cryptorl.data import MarketData
from cryptorl.llm import ConfidenceGenerator

# Collect market data
market_data = MarketData()
market_data.collect_historical(
    symbols=["BTCUSDT", "ETHUSDT"],
    interval="4h",
    days=90
)

# Generate confidence vectors
confidence_gen = ConfidenceGenerator()
confidence_gen.generate_batch(
    symbols=["BTCUSDT", "ETHUSDT"],
    start_date="2024-01-01",
    end_date="2024-03-31"
)
```

#### Data Fusion
```python
from cryptorl.data import DataFusion

# Combine market data with confidence vectors
fusion = DataFusion()
training_data = fusion.create_training_dataset(
    symbols=["BTCUSDT", "ETHUSDT"],
    include_confidence=True,
    technical_indicators=["RSI", "MACD", "BB", "ATR"],
    lookback_window=20
)
```

### 3. Algorithm Selection

#### PPO (Proximal Policy Optimization)
```python
from stable_baselines3 import PPO
from cryptorl.rl import TradingEnvironment

env = TradingEnvironment(symbols=["BTCUSDT"])

# PPO Configuration
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tensorboard_logs/ppo/"
)

# Train model
model.learn(total_timesteps=100000)
model.save("models/ppo_btc_model")
```

#### SAC (Soft Actor-Critic)
```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    tensorboard_log="./tensorboard_logs/sac/"
)

model.learn(total_timesteps=100000)
```

#### TD3 (Twin Delayed DDPG)
```python
from stable_baselines3 import TD3

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=100,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    verbose=1,
    tensorboard_log="./tensorboard_logs/td3/"
)
```

### 4. Mamba Architecture

#### Experimental Mamba Setup
```python
from cryptorl.rl import MambaExploration

# Create Mamba-based environment
mamba_env = MambaExploration.create_environment(
    symbols=["BTCUSDT"],
    sequence_length=50,
    d_model=256,
    n_layers=4,
    use_confidence=True
)

# Mamba model configuration
mamba_config = {
    "d_model": 256,
    "n_layers": 4,
    "vocab_size": 1000,
    "rms_norm": True,
    "residual_in_fp32": True,
    "fused_add_norm": True,
    "pad_vocab_size_multiple": 8
}

model = MambaExploration.train_mamba_agent(
    env=mamba_env,
    config=mamba_config,
    learning_rate=1e-4,
    total_timesteps=50000
)
```

### 5. Training Configuration

#### Hyperparameter Tuning
```python
from cryptorl.rl import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    algorithm="PPO",
    env=TradingEnvironment(symbols=["BTCUSDT"])
)

# Define search space
search_space = {
    "learning_rate": [1e-5, 1e-3],
    "n_steps": [512, 2048, 4096],
    "batch_size": [32, 64, 128],
    "gamma": [0.9, 0.99, 0.999],
    "ent_coef": [0.001, 0.01, 0.1]
}

# Run optimization
best_params = optimizer.optimize(
    search_space=search_space,
    n_trials=50,
    metric="sharpe_ratio"
)
```

#### Multi-symbol Training
```python
from cryptorl.rl import MultiSymbolTrainer

trainer = MultiSymbolTrainer(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"],
    algorithm="PPO",
    total_timesteps=500000,
    confidence_data=True
)

# Train across all symbols
trainer.train_all()

# Get performance summary
summary = trainer.get_performance_summary()
print(summary)
```

### 6. Advanced Training Techniques

#### Curriculum Learning
```python
from cryptorl.rl import CurriculumTrainer

trainer = CurriculumTrainer(
    base_env=TradingEnvironment(symbols=["BTCUSDT"])
)

# Define curriculum stages
stages = [
    {"volatility": 0.01, "trend_strength": 0.5},
    {"volatility": 0.02, "trend_strength": 0.7},
    {"volatility": 0.03, "trend_strength": 0.9}
]

trainer.train_curriculum(
    stages=stages,
    episodes_per_stage=1000
)
```

#### Transfer Learning
```python
from cryptorl.rl import TransferTrainer

# Train on source symbol
source_trainer = TransferTrainer(
    source_symbol="BTCUSDT",
    target_symbol="ETHUSDT"
)

# Transfer knowledge
source_trainer.train_source(total_timesteps=100000)
source_trainer.transfer_to_target(
    target_timesteps=50000,
    freeze_layers=["feature_extractor"]
)
```

#### Ensemble Methods
```python
from cryptorl.rl import EnsembleTrainer

# Create ensemble of different algorithms
ensemble = EnsembleTrainer(
    algorithms=["PPO", "SAC", "TD3"],
    env=TradingEnvironment(symbols=["BTCUSDT"])
)

# Train ensemble
ensemble.train_all(total_timesteps=100000)

# Get weighted predictions
predictions = ensemble.predict(
    market_state=current_state,
    weights=[0.4, 0.3, 0.3]  # PPO, SAC, TD3 weights
)
```

### 7. Evaluation and Validation

#### Backtesting
```python
from cryptorl.rl import BacktestEngine

# Create backtest engine
backtest = BacktestEngine(
    model_path="models/ppo_btc_model",
    test_data="data/test_data.csv",
    initial_balance=10000
)

# Run backtest
results = backtest.run(
    start_date="2024-04-01",
    end_date="2024-06-30"
)

# Performance metrics
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

#### Walk-Forward Analysis
```python
from cryptorl.rl import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(
    model_type="PPO",
    symbols=["BTCUSDT"],
    train_days=60,
    test_days=10,
    step_days=5
)

# Run walk-forward analysis
results = analyzer.run_analysis(
    total_timesteps=100000,
    n_splits=10
)

# Plot results
analyzer.plot_results(results)
```

### 8. Model Monitoring

#### Real-time Performance Tracking
```python
from cryptorl.rl import ModelMonitor

monitor = ModelMonitor(
    model_path="models/ppo_btc_model",
    symbols=["BTCUSDT"]
)

# Start monitoring
monitor.start_monitoring(
    check_interval=3600,  # Check every hour
    metrics=["sharpe", "drawdown", "win_rate"],
    alert_thresholds={
        "sharpe_min": 0.5,
        "drawdown_max": 0.15
    }
)
```

#### Model Registry
```python
from cryptorl.rl import ModelRegistry

registry = ModelRegistry()

# Register model
registry.register_model(
    model_path="models/ppo_btc_v2",
    metadata={
        "algorithm": "PPO",
        "symbols": ["BTCUSDT"],
        "sharpe_ratio": 1.8,
        "max_drawdown": 0.12,
        "training_date": "2024-06-15"
    }
)

# Get best model for symbol
best_model = registry.get_best_model("BTCUSDT")
```

### 9. Production Deployment

#### Model Serving
```python
from cryptorl.rl import ModelServer

# Start model server
server = ModelServer(
    model_path="models/ppo_btc_production",
    host="0.0.0.0",
    port=8000
)

server.start()
```

#### A/B Testing
```python
from cryptorl.rl import ABTesting

test = ABTesting(
    model_a="models/ppo_v1",
    model_b="models/ppo_v2",
    split_ratio=0.5
)

# Run test
test_results = test.run_test(
    duration_days=7,
    symbols=["BTCUSDT"]
)

# Analyze results
winner = test.declare_winner()
```

## Training Best Practices

### 1. Data Quality
- Use high-quality historical data
- Include market regime changes
- Validate confidence vectors
- Check for data gaps

### 2. Hyperparameter Tuning
- Start with default parameters
- Use Bayesian optimization
- Validate on out-of-sample data
- Monitor training stability

### 3. Risk Management
- Implement position limits
- Use stop-losses
- Monitor drawdown
- Test with paper trading

### 4. Monitoring
- Track Sharpe ratio
- Monitor maximum drawdown
- Check win rate trends
- Validate with walk-forward

### 5. Continuous Learning
- Retrain models regularly
- Adapt to market changes
- Update confidence vectors
- Monitor model drift

## Common Training Issues

### 1. Overfitting
```python
# Early stopping
from stable_baselines3.common.callbacks import EarlyStoppingCallback

early_stop = EarlyStoppingCallback(
    eval_freq=1000,
    patience=5,
    verbose=1
)

model.learn(
    total_timesteps=100000,
    callback=early_stop
)
```

### 2. Training Instability
```python
# Gradient clipping
model = PPO(
    "MlpPolicy",
    env,
    max_grad_norm=0.5,
    clip_range=0.1,
    learning_rate=linear_schedule(3e-4)
)
```

### 3. Poor Convergence
```python
# Adjust reward scaling
env = TradingEnvironment(
    symbols=["BTCUSDT"],
    reward_scaling=0.01,
    normalize_rewards=True
)
```