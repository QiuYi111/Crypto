#!/usr/bin/env python3
"""
Phase 3 Demo: Multi-market RL training with Mamba architecture (Fixed)
"""

import asyncio
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import logging

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cryptorl.rl.environment import CryptoTradingEnvironment, MultiSymbolTradingEnvironment
from src.cryptorl.rl.agent import CryptoRLAgent
from src.cryptorl.rl.mamba_exploration import MambaExplorer
from src.cryptorl.config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_training_data():
    """Create sample training data for demo."""
    
    # Generate synthetic data for BTC, ETH, SOL
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    data_list = []
    for symbol in symbols:
        for date in dates:
            # Generate realistic price patterns
            base_price = {'BTCUSDT': 30000, 'ETHUSDT': 2000, 'SOLUSDT': 30}[symbol]
            price_noise = np.random.normal(0, 0.02)
            
            data_list.append({
                'symbol': symbol,
                'date': date,
                'open': base_price * (1 + price_noise),
                'high': base_price * (1 + price_noise + abs(np.random.normal(0, 0.01))),
                'low': base_price * (1 + price_noise - abs(np.random.normal(0, 0.01))),
                'close': base_price * (1 + price_noise),
                'volume': np.random.randint(1000000, 10000000),
                'close_norm': np.random.normal(0, 1),
                'volume_norm': np.random.normal(0, 1),
                'price_change': np.random.normal(0, 0.02),
                'rsi_norm': np.random.normal(0.5, 0.2),
                'macd_norm': np.random.normal(0, 0.1),
                'volume_ratio_norm': np.random.normal(1, 0.3),
                # Confidence vectors from LLM
                'confidence_fundamentals': np.random.beta(2, 2),
                'confidence_industry': np.random.beta(2, 2),
                'confidence_geopolitics': np.random.beta(2, 2),
                'confidence_macro': np.random.beta(2, 2),
                'confidence_technical': np.random.beta(2, 2),
                'confidence_regulatory': np.random.beta(2, 2),
                'confidence_innovation': np.random.beta(2, 2),
                'confidence_overall': np.random.beta(2, 2)
            })
    
    return pd.DataFrame(data_list)


async def test_mamba_exploration():
    """Test Mamba architecture exploration."""
    
    logger.info("=== Testing Mamba Architecture Exploration ===")
    
    settings = Settings()
    explorer = MambaExplorer(settings)
    
    # Test different architectures
    observation_dim = 20  # Based on our environment features
    action_dim = 1
    
    # Quick exploration
    benchmarks = explorer.explore_architectures(
        observation_dim=observation_dim,
        action_dim=action_dim,
        sequence_lengths=[30, 60],
        batch_sizes=[1, 8],
        hidden_dims=[128, 256],
        num_layers=[2, 4]
    )
    
    # Compare with baselines
    baseline_benchmarks = explorer.compare_with_baselines(
        observation_dim=observation_dim,
        action_dim=action_dim,
        sequence_length=30,
        batch_size=8
    )
    
    # Export results
    explorer.export_benchmark_results(benchmarks + baseline_benchmarks, "mamba_phase3_benchmarks.csv")
    
    # Recommend best config
    best_config = explorer.recommend_best_config(benchmarks + baseline_benchmarks)
    
    logger.info(f"Best configuration: {best_config.model_name}")
    logger.info(f"Parameters: {best_config.parameters:,}")
    logger.info(f"Inference time: {best_config.inference_time:.4f}s")
    
    return best_config


async def test_multi_market_training():
    """Test multi-market training."""
    
    logger.info("=== Testing Multi-Market Training ===")
    
    settings = Settings()
    
    # Create sample training data
    training_data = await create_sample_training_data()
    
    # Create environments for each symbol
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    environments = []
    
    for symbol in symbols:
        symbol_data = training_data[training_data['symbol'] == symbol]
        
        env = CryptoTradingEnvironment(
            settings=settings,
            data=symbol_data,
            symbols=[symbol],
            initial_balance=settings.rl_initial_balance,
            max_position_size=settings.rl_max_position_size,
            trading_fee=settings.rl_trading_fee,
            max_episode_length=settings.rl_max_episode_length
        )
        environments.append(env)
    
    # Create agent
    sample_env = environments[0]
    agent = CryptoRLAgent(
        settings=settings,
        observation_space=sample_env.observation_space,
        action_space=sample_env.action_space,
        model_type="mamba",
        learning_rate=settings.rl_learning_rate
    )
    
    # Quick training test
    logger.info("Starting quick training test...")
    
    episode_rewards = []
    for episode in range(10):  # Quick test with 10 episodes
        env = environments[episode % len(environments)]
        episode_metrics = agent.train_episode(env)
        episode_rewards.append(episode_metrics['episode_reward'])
        
        if episode % 3 == 0:
            logger.info(f"Episode {episode}: Reward={episode_metrics['episode_reward']:.2f}")
    
    # Evaluate performance
    eval_results = []
    for env in environments:
        eval_metrics = agent.evaluate(env, num_episodes=3)
        eval_results.append({
            'symbol': env.symbols[0],
            **eval_metrics
        })
    
    logger.info("\n=== Evaluation Results ===")
    for result in eval_results:
        logger.info(f"{result['symbol']}: Mean Reward={result['mean_reward']:.2f}, "
                   f"Win Rate={result['win_rate']:.2f}")
    
    return eval_results


async def test_multi_symbol_environment():
    """Test multi-symbol environment."""
    
    logger.info("=== Testing Multi-Symbol Environment ===")
    
    settings = Settings()
    training_data = await create_sample_training_data()
    
    # Create multi-symbol environment
    multi_env = MultiSymbolTradingEnvironment(
        settings=settings,
        data=training_data,
        symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        initial_balance=settings.rl_initial_balance,
        max_position_size=settings.rl_max_position_size,
        trading_fee=settings.rl_trading_fee,
        max_episode_length=settings.rl_max_episode_length
    )
    
    # Test environment
    observation, info = multi_env.reset()
    logger.info(f"Observation shape: {observation.shape}")
    logger.info(f"Action space: {multi_env.action_space}")
    
    # Run a few steps
    total_reward = 0
    for step in range(5):
        action = np.array([np.random.uniform(-1, 1)])  # Random action
        observation, reward, terminated, truncated, info = multi_env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    logger.info(f"Multi-symbol environment test completed. Total reward: {total_reward:.2f}")
    
    return multi_env


async def run_phase3_demo():
    """Run complete Phase 3 demonstration."""
    
    logger.info("Starting Phase 3 Demo: RL Training with Mamba Architecture")
    
    try:
        # Test 1: Mamba Architecture Exploration
        best_config = await test_mamba_exploration()
        
        # Test 2: Multi-Market Training
        eval_results = await test_multi_market_training()
        
        # Test 3: Multi-Symbol Environment
        multi_env = await test_multi_symbol_environment()
        
        logger.info("\n=== Phase 3 Demo Summary ===")
        logger.info("✅ Mamba architecture exploration completed")
        logger.info("✅ Multi-market training tested")
        logger.info("✅ Multi-symbol environment validated")
        logger.info("✅ Phase 3 implementation ready for full training")
        
        return {
            'best_config': best_config,
            'eval_results': eval_results,
            'multi_env': multi_env
        }
        
    except Exception as e:
        logger.error(f"Phase 3 demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_phase3_demo())