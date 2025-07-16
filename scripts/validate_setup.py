#!/usr/bin/env python3
"""
Validation script to test the complete CryptoRL system.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cryptorl.config.settings import settings
from cryptorl.data.influxdb_client import InfluxDBClient
from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator
from cryptorl.rl.agent import CryptoRLAgent
from cryptorl.rl.environment import CryptoTradingEnvironment
from cryptorl.trading.execution import TradingExecutor
from cryptorl.rl.mamba_exploration import MambaExplorer


async def validate_database():
    """Validate database connections."""
    print("🔍 Validating database connections...")
    
    try:
        influx_client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
            bucket=settings.influxdb_bucket
        )
        
        health = await influx_client.health_check()
        if health["status"] == "healthy":
            print("✅ InfluxDB: Connected")
            return True
        else:
            print(f"❌ InfluxDB: {health}")
            return False
            
    except Exception as e:
        print(f"❌ InfluxDB: {e}")
        return False


async def validate_llm_system():
    """Validate LLM confidence generation system."""
    print("🧠 Validating LLM system...")
    
    try:
        from cryptorl.data.influxdb_client import InfluxDBClient
        influx_client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
            bucket=settings.influxdb_bucket
        )
        
        generator = ConfidenceVectorGenerator(settings, influx_client)
        health = await generator.health_check()
        
        if health["status"] == "healthy":
            print("✅ LLM System: Healthy")
            return True
        else:
            print(f"⚠️  LLM System: {health}")
            return False
            
    except Exception as e:
        print(f"❌ LLM System: {e}")
        return False


async def validate_trading_system():
    """Validate trading system."""
    print("⚙️  Validating trading system...")
    
    try:
        executor = TradingExecutor(settings)
        health = await executor.health_check()
        
        if health["status"] == "healthy":
            print("✅ Trading System: Healthy")
            return True
        else:
            print(f"❌ Trading System: {health}")
            return False
            
    except Exception as e:
        print(f"❌ Trading System: {e}")
        return False


async def validate_mamba_exploration():
    """Validate Mamba model exploration."""
    print("🔍 Validating Mamba exploration...")
    
    try:
        explorer = MambaExplorer(settings)
        
        # Quick benchmark test
        benchmarks = explorer.compare_with_baselines(
            observation_dim=10,
            action_dim=1,
            sequence_length=10,
            batch_size=1
        )
        
        if benchmarks:
            print("✅ Mamba Exploration: Working")
            print(f"   - Models tested: {len(benchmarks)}")
            print(f"   - Best accuracy: {max(b.accuracy for b in benchmarks):.3f}")
            return True
        else:
            print("❌ Mamba Exploration: No results")
            return False
            
    except Exception as e:
        print(f"❌ Mamba Exploration: {e}")
        return False


async def validate_environment():
    """Validate RL environment."""
    print("🎮 Validating RL environment...")
    
    try:
        # Create sample data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        sample_data = pd.DataFrame({
            'date': dates,
            'symbol': 'BTCUSDT',
            'open': np.random.uniform(30000, 40000, 100),
            'high': np.random.uniform(31000, 41000, 100),
            'low': np.random.uniform(29000, 39000, 100),
            'close': np.random.uniform(30000, 40000, 100),
            'volume': np.random.uniform(1000, 10000, 100),
            'confidence_fundamentals': np.random.uniform(0.4, 0.6, 100),
            'confidence_industry': np.random.uniform(0.4, 0.6, 100),
            'confidence_geopolitics': np.random.uniform(0.4, 0.6, 100),
            'confidence_macro': np.random.uniform(0.4, 0.6, 100),
            'confidence_technical': np.random.uniform(0.4, 0.6, 100),
            'confidence_regulatory': np.random.uniform(0.4, 0.6, 100),
            'confidence_innovation': np.random.uniform(0.4, 0.6, 100),
            'confidence_overall': np.random.uniform(0.4, 0.6, 100),
            'close_norm': np.random.uniform(-1, 1, 100),
            'volume_norm': np.random.uniform(-1, 1, 100),
            'rsi_norm': np.random.uniform(0, 1, 100),
            'macd_norm': np.random.uniform(-1, 1, 100),
            'volume_ratio_norm': np.random.uniform(0.5, 2, 100)
        })
        
        env = CryptoTradingEnvironment(
            settings=settings,
            data=sample_data,
            symbols=["BTCUSDT"],
            initial_balance=10000,
            max_position_size=0.1,
            trading_fee=0.001,
            max_episode_length=30
        )
        
        # Test environment
        observation, info = env.reset()
        assert observation.shape[0] == env.observation_space.shape[0]
        
        # Test step
        action = [0.5]
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✅ RL Environment: Working")
        print(f"   - Observation shape: {observation.shape}")
        print(f"   - Action space: {env.action_space}")
        
        return True
        
    except Exception as e:
        print(f"❌ RL Environment: {e}")
        return False


async def validate_dependencies():
    """Validate critical dependencies."""
    print("📦 Validating dependencies...")
    
    dependencies = {
        "torch": None,
        "numpy": None,
        "pandas": None,
        "gymnasium": None,
        "transformers": None,
        "mamba_ssm": None
    }
    
    results = {}
    
    for dep in dependencies:
        try:
            if dep == "mamba_ssm":
                from mamba_ssm import Mamba
                results[dep] = "✅ Available"
            elif dep == "gymnasium":
                import gymnasium as gym
                results[dep] = "✅ Available"
            else:
                __import__(dep)
                results[dep] = "✅ Available"
        except ImportError:
            results[dep] = "❌ Missing"
    
    for dep, status in results.items():
        print(f"   {dep}: {status}")
    
    return all("Available" in status for status in results.values())


async def validate_configuration():
    """Validate configuration settings."""
    print("⚙️  Validating configuration...")
    
    required_vars = [
        "BINANCE_API_KEY",
        "BINANCE_SECRET_KEY",
        "INFLUXDB_TOKEN",
        "SERPAPI_KEY"
    ]
    
    missing = []
    for var in required_vars:
        value = getattr(settings, var.lower(), None)
        if not value:
            missing.append(var)
    
    if missing:
        print("❌ Missing configuration:")
        for var in missing:
            print(f"   - {var}")
        return False
    else:
        print("✅ Configuration complete")
        return True


async def main():
    """Run complete system validation."""
    print("🚀 CryptoRL System Validation")
    print("=" * 50)
    
    validations = [
        ("Configuration", validate_configuration),
        ("Dependencies", validate_dependencies),
        ("Database", validate_database),
        ("LLM System", validate_llm_system),
        ("Trading System", validate_trading_system),
        ("Mamba Exploration", validate_mamba_exploration),
        ("RL Environment", validate_environment)
    ]
    
    results = []
    
    for name, validator in validations:
        print(f"\n{name}...")
        try:
            success = await validator()
            results.append((name, success))
        except Exception as e:
            results.append((name, False))
            print(f"   ❌ Exception: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20} {status}")
        if success:
            passed += 1
    
    print(f"\n{passed}/{len(results)} validations passed")
    
    if passed == len(results):
        print("🎉 System is ready for use!")
        return True
    else:
        print("⚠️  Some validations failed. Check configuration and dependencies.")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = asyncio.run(main())
    sys.exit(0 if success else 1)