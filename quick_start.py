#!/usr/bin/env python3
"""
Quick start demo for CryptoRL agent - runs without external dependencies
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptorl.config.settings import settings
from cryptorl.rl.environment import CryptoTradingEnvironment
from cryptorl.rl.agent import CryptoRLAgent
from cryptorl.data.fusion import DataFusionEngine

def create_sample_data():
    """Create sample market data for testing."""
    
    # Generate 30 days of synthetic BTC data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # Simulate BTC price movement
    np.random.seed(42)  # Reproducible results
    prices = 40000 + np.cumsum(np.random.randn(30) * 1000)
    
    # Create OHLCV data with required features
    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'BTCUSDT',
        'open': prices,
        'high': prices + np.abs(np.random.randn(30) * 200),
        'low': prices - np.abs(np.random.randn(30) * 200),
        'close': prices + np.random.randn(30) * 100,
        'volume': np.random.randint(10000, 100000, 30),
        'close_norm': (prices + np.random.randn(30) * 100) / prices[0],
        'volume_norm': np.random.rand(30),
        'price_change': np.random.randn(30) * 0.02,
        'rsi_norm': np.random.rand(30),
        'macd_norm': np.random.randn(30) * 0.1,
        'volume_ratio_norm': np.random.rand(30),
        'confidence_fundamentals': np.random.rand(30),
        'confidence_industry': np.random.rand(30),
        'confidence_geopolitics': np.random.rand(30),
        'confidence_macro': np.random.rand(30),
        'confidence_technical': np.random.rand(30),
        'confidence_regulatory': np.random.rand(30),
        'confidence_innovation': np.random.rand(30),
        'confidence_overall': np.random.rand(30)
    })
    
    return data

def run_demo():
    """Run a complete demo without external dependencies."""
    
    print("🚀 CryptoRL Quick Start Demo")
    print("=" * 50)
    
    # Load settings
    print("✅ Loading configuration...")
    print(f"   Project: {settings.project_name}")
    print(f"   Symbols: {settings.trading_symbols}")
    print(f"   Testnet: {settings.binance_testnet}")
    
    # Create sample data
    print("\n📊 Creating sample data...")
    sample_data = create_sample_data()
    print(f"   Data shape: {sample_data.shape}")
    print(f"   Date range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
    
    # Create environment
    print("\n🎯 Creating trading environment...")
    env = CryptoTradingEnvironment(
        settings=settings,
        data=sample_data,
        symbols=['BTCUSDT'],
        initial_balance=10000.0,
        max_position_size=1.0,
        trading_fee=0.001
    )
    
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Create agent
    print("\n🤖 Creating RL agent...")
    agent = CryptoRLAgent(
        settings=settings,
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_type="transformer",  # Use transformer instead of mamba
        learning_rate=1e-3
    )
    
    print(f"   Model type: Transformer")
    print(f"   Parameters: {sum(p.numel() for p in agent.policy.parameters())}")
    
    # Run simple training
    print("\n🎓 Running quick training...")
    total_reward = 0
    obs, _ = env.reset()
    
    for step in range(10):
        action, _ = agent.select_action(obs, training=False)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            obs, _ = env.reset()
    
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final balance: {env.balance:.2f}")
    
    # Save results
    print("\n💾 Saving results...")
    results = {
        'final_balance': env.balance,
        'total_reward': total_reward,
        'trades_executed': len(env.episode_history),
        'model_type': 'transformer'
    }
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame([results]).to_csv("results/quick_start_results.csv", index=False)
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Update .env with your real API keys")
    print("2. Set BINANCE_TESTNET=true for safe testing")
    print("3. Run: python scripts/collect_data.py")
    print("4. Run: python scripts/phase3_demo.py")

if __name__ == "__main__":
    run_demo()