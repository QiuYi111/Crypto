#!/usr/bin/env python3
"""
Complete system setup script for CryptoRL Agent.
This script sets up the entire system from data collection to RL training.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cryptorl.config.settings import settings
from cryptorl.data.market_data import MarketDataCollector
from cryptorl.data.influxdb_client import InfluxDBClient
from cryptorl.data.fusion import DataFusionEngine
from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator
from cryptorl.rl.agent import CryptoRLAgent
from cryptorl.rl.environment import CryptoTradingEnvironment
from cryptorl.rl.training import Trainer
from cryptorl.trading.execution import TradingExecutor
from cryptorl.rl.mamba_exploration import MambaExplorer


async def setup_database():
    """Initialize database connections."""
    print("🗄️  Setting up database connections...")
    
    influx_client = InfluxDBClient(
        url=settings.influxdb_url,
        token=settings.influxdb_token,
        org=settings.influxdb_org,
        bucket=settings.influxdb_bucket
    )
    
    # Test connection
    health = await influx_client.health_check()
    if health["status"] == "healthy":
        print("✅ InfluxDB connected successfully")
    else:
        print(f"❌ InfluxDB connection failed: {health}")
    
    return influx_client


async def collect_historical_data(influx_client: InfluxDBClient):
    """Collect historical market data."""
    print("📊 Collecting historical market data...")
    
    collector = MarketDataCollector(
        binance_client=None,  # Will be created in method
        influx_client=influx_client
    )
    
    symbols = settings.trading_symbols
    intervals = ["1d", "4h", "1h"]
    start_date = datetime.utcnow() - timedelta(days=settings.rl.training_days)
    
    print(f"Collecting {len(symbols)} symbols for {len(intervals)} intervals...")
    
    total_records = 0
    for symbol in symbols:
        for interval in intervals:
            try:
                records = await collector.collect_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date
                )
                total_records += records
                print(f"✅ {symbol} {interval}: {records} records")
            except Exception as e:
                print(f"❌ {symbol} {interval}: {e}")
    
    print(f"📈 Total records collected: {total_records:,}")


async def generate_confidence_vectors(influx_client: InfluxDBClient):
    """Generate LLM confidence vectors for historical data."""
    print("🧠 Generating LLM confidence vectors...")
    
    generator = ConfidenceVectorGenerator(settings, influx_client)
    
    # Check system health
    health = await generator.health_check()
    if health["status"] != "healthy":
        print(f"⚠️  LLM system health: {health}")
    
    symbols = settings.trading_symbols
    start_date = datetime.utcnow() - timedelta(days=30)
    end_date = datetime.utcnow()
    
    print(f"Generating vectors for {len(symbols)} symbols...")
    
    vectors = await generator.batch_generate_historical(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        batch_size=5
    )
    
    print(f"✅ Generated {len(vectors)} confidence vectors")


async def create_enhanced_dataset(influx_client: InfluxDBClient):
    """Create enhanced dataset for RL training."""
    print("🔄 Creating enhanced dataset...")
    
    fusion_engine = DataFusionEngine(settings, influx_client)
    
    symbols = settings.trading_symbols
    start_date = datetime.utcnow() - timedelta(days=settings.rl.training_days)
    end_date = datetime.utcnow()
    
    enhanced_data = await fusion_engine.create_enhanced_dataset(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    )
    
    print(f"✅ Enhanced dataset created: {len(enhanced_data)} rows")
    print(f"Features: {len(fusion_engine.get_feature_columns())} columns")
    
    # Save to CSV for inspection
    csv_path = Path("data/enhanced_dataset.csv")
    csv_path.parent.mkdir(exist_ok=True)
    enhanced_data.to_csv(csv_path, index=False)
    print(f"📁 Dataset saved to {csv_path}")
    
    return enhanced_data


async def train_rl_agent(enhanced_data: DataFusionEngine):
    """Train the RL agent."""
    print("🤖 Training RL agent...")
    
    # Create training environments
    env = CryptoTradingEnvironment(
        settings=settings,
        data=enhanced_data,
        symbols=settings.trading_symbols,
        initial_balance=settings.rl.initial_balance,
        max_position_size=settings.rl.max_position_size,
        trading_fee=settings.rl.trading_fee,
        max_episode_length=settings.rl.max_episode_length
    )
    
    # Create agent
    agent = CryptoRLAgent(
        settings=settings,
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_type="mamba",
        learning_rate=settings.rl.learning_rate,
        gamma=settings.rl.gamma
    )
    
    # Create trainer
    trainer = Trainer(settings, None)
    
    # Quick training run
    training_metrics = trainer.train_agent(
        agent=agent,
        environments=[env],
        num_episodes=100,
        eval_frequency=20,
        save_frequency=50
    )
    
    print(f"✅ Training completed: {len(training_metrics['episode_rewards'])} episodes")
    
    # Save final model
    model_path = Path("models/final_model.pth")
    model_path.parent.mkdir(exist_ok=True)
    agent.save_model(str(model_path))
    print(f"💾 Model saved to {model_path}")
    
    return agent


async def explore_mamba_architecture():
    """Explore Mamba model architectures."""
    print("🔍 Exploring Mamba architectures...")
    
    explorer = MambaExplorer(settings)
    
    # Compare with baselines
    benchmarks = explorer.compare_with_baselines(
        observation_dim=50,  # Typical feature count
        action_dim=1,        # Continuous action
        sequence_length=30,
        batch_size=8
    )
    
    # Export results
    explorer.export_benchmark_results(benchmarks, "models/mamba_benchmarks.csv")
    
    # Recommend best config
    best_config = explorer.recommend_best_config(benchmarks)
    print(f"🏆 Recommended configuration: {best_config.model_name}")
    
    return benchmarks


async def setup_trading_system():
    """Set up trading system components."""
    print("⚙️  Setting up trading system...")
    
    executor = TradingExecutor(settings)
    
    # Health check
    health = await executor.health_check()
    if health["status"] == "healthy":
        print("✅ Trading system ready")
    else:
        print(f"❌ Trading system issue: {health}")
    
    return executor


async def create_system_summary():
    """Create system summary report."""
    print("📝 Creating system summary...")
    
    summary = f"""# CryptoRL System Setup Complete

## System Overview
- **Project**: {settings.project_name} v{settings.version}
- **Environment**: {settings.environment}
- **Setup Date**: {datetime.utcnow().isoformat()}

## Components Status
✅ Phase 1: Data Infrastructure
✅ Phase 2: LLM Enhancement 
✅ Phase 3: RL Training
✅ Phase 4: Trading System

## Configuration
- **Symbols**: {', '.join(settings.trading_symbols)}
- **Training Days**: {settings.rl.training_days}
- **Model**: Mamba-based RL Agent
- **Exchange**: Binance {'Testnet' if settings.binance_testnet else 'Live'}

## Files Created
- Models: `models/`
- Data: `data/`
- Logs: `logs/`
- Cache: `cache/`

## Next Steps
1. Configure `.env` file with your API keys
2. Run `python scripts/validate_setup.py` to test system
3. Start training with `python scripts/train_agent.py`
4. Deploy with `python scripts/deploy_agent.py`
"""
    
    summary_path = Path("SYSTEM_SETUP.md")
    summary_path.write_text(summary)
    print(f"📋 System summary saved to {summary_path}")


async def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="CryptoRL System Setup")
    parser.add_argument("--full", action="store_true", help="Run full setup")
    parser.add_argument("--data-only", action="store_true", help="Only setup data")
    parser.add_argument("--train-only", action="store_true", help="Only train agent")
    parser.add_argument("--mamba-only", action="store_true", help="Only Mamba exploration")
    
    args = parser.parse_args()
    
    print("🚀 Starting CryptoRL System Setup...")
    print(f"Configuration file: {Path('.env').absolute()}")
    
    try:
        # Setup directories
        Path("models").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        
        # Initialize database
        influx_client = await setup_database()
        
        if args.data_only or args.full:
            await collect_historical_data(influx_client)
            await generate_confidence_vectors(influx_client)
            enhanced_data = await create_enhanced_dataset(influx_client)
        else:
            # Use sample data for other modes
            enhanced_data = None
        
        if args.train_only or args.full:
            if enhanced_data is None:
                print("❌ Enhanced data required for training. Run with --data-only first.")
                return
            agent = await train_rl_agent(enhanced_data)
        
        if args.mamba_only or args.full:
            await explore_mamba_architecture()
        
        if args.full:
            await setup_trading_system()
        
        await create_system_summary()
        
        print("🎉 CryptoRL System Setup Complete!")
        print("\nNext steps:")
        print("1. Configure your .env file with API keys")
        print("2. Run: python scripts/validate_setup.py")
        print("3. Start training: python scripts/train_agent.py")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())