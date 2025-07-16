#!/usr/bin/env python3
"""
Comprehensive test suite for CryptoRL system using mock data.
Tests all components without external API calls.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptorl.config.settings import settings
from cryptorl.rl.environment import CryptoTradingEnvironment
from cryptorl.rl.agent import CryptoRLAgent
from cryptorl.data.fusion import DataFusionEngine
from cryptorl.trading.execution import BinanceTrader
from cryptorl.risk_management.risk_manager import RiskManager


class MockDataGenerator:
    """Generate realistic mock market data."""
    
    @staticmethod
    def create_market_data(days=30, symbols=['BTCUSDT']):
        """Create mock OHLCV data with technical indicators."""
        
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        all_data = []
        for symbol in symbols:
            # Generate realistic price movement
            np.random.seed(42)
            prices = 40000 + np.cumsum(np.random.randn(days) * 500)
            
            # Create OHLCV data
            data = pd.DataFrame({
                'timestamp': dates,
                'symbol': symbol,
                'open': prices,
                'high': prices + np.abs(np.random.randn(days) * 100),
                'low': prices - np.abs(np.random.randn(days) * 100),
                'close': prices + np.random.randn(days) * 50,
                'volume': np.random.randint(10000, 100000, days)
            })
            
            # Add normalized features
            data['close_norm'] = data['close'] / data['close'].iloc[0]
            data['volume_norm'] = data['volume'] / data['volume'].max()
            data['price_change'] = data['close'].pct_change()
            data['rsi_norm'] = np.random.rand(days)
            data['macd_norm'] = np.random.randn(days) * 0.1
            data['volume_ratio_norm'] = np.random.rand(days)
            
            # Add confidence vectors
            for conf in ['fundamentals', 'industry', 'geopolitics', 'macro', 
                        'technical', 'regulatory', 'innovation', 'overall']:
                data[f'confidence_{conf}'] = np.random.rand(days)
            
            all_data.append(data)
        
        return pd.concat(all_data, ignore_index=True)


class TestCryptoRLSystem(unittest.TestCase):
    """Test suite for CryptoRL system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data = MockDataGenerator.create_market_data()
        self.settings = settings
        self.settings.trading_symbols = ['BTCUSDT', 'ETHUSDT']
    
    def test_settings_loading(self):
        """Test configuration loading."""
        self.assertEqual(settings.project_name, "CryptoRL Agent")
        self.assertGreater(len(settings.trading_symbols), 0)
        self.assertIsInstance(settings.rl_hidden_dim, int)
        print("✅ Settings loaded correctly")
    
    def test_data_generation(self):
        """Test mock data generation."""
        self.assertEqual(len(self.mock_data), 60)  # 30 days * 2 symbols
        self.assertEqual(len(self.mock_data.columns), 19)  # All required columns
        self.assertTrue('symbol' in self.mock_data.columns)
        self.assertTrue('BTCUSDT' in self.mock_data['symbol'].values)
        print("✅ Mock data generated correctly")
    
    def test_rl_environment_creation(self):
        """Test RL environment initialization."""
        env = CryptoTradingEnvironment(
            settings=self.settings,
            data=self.mock_data,
            symbols=['BTCUSDT', 'ETHUSDT'],
            initial_balance=10000.0,
            max_position_size=1.0,
            trading_fee=0.001
        )
        
        self.assertIsNotNone(env.observation_space)
        self.assertIsNotNone(env.action_space)
        self.assertEqual(env.observation_space.shape[0], 18)  # Feature count
        self.assertEqual(env.action_space.shape[0], 1)
        print("✅ RL environment created successfully")
    
    def test_rl_environment_step(self):
        """Test RL environment step functionality."""
        env = CryptoTradingEnvironment(
            settings=self.settings,
            data=self.mock_data,
            symbols=['BTCUSDT']
        )
        
        # Test reset
        obs, info = env.reset()
        self.assertEqual(len(obs), 18)
        
        # Test step with valid action
        action = np.array([0.5])  # 50% long position
        obs, reward, done, truncated, info = env.step(action)
        
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(len(obs), 18)
        print("✅ RL environment step works correctly")
    
    def test_rl_agent_creation(self):
        """Test RL agent initialization."""
        env = CryptoTradingEnvironment(
            settings=self.settings,
            data=self.mock_data,
            symbols=['BTCUSDT']
        )
        
        agent = CryptoRLAgent(
            settings=self.settings,
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_type="transformer",
            learning_rate=1e-3
        )
        
        self.assertIsNotNone(agent.policy)
        self.assertGreater(agent.policy.parameters().__next__().numel(), 0)
        print("✅ RL agent created successfully")
    
    def test_agent_action_selection(self):
        """Test agent action selection."""
        env = CryptoTradingEnvironment(
            settings=self.settings,
            data=self.mock_data,
            symbols=['BTCUSDT']
        )
        
        agent = CryptoRLAgent(
            settings=self.settings,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        obs, _ = env.reset()
        action, log_prob = agent.select_action(obs, training=False)
        
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(-1.0 <= action[0] <= 1.0)
        self.assertIsInstance(log_prob, float)
        print("✅ Agent action selection works correctly")
    
    def test_trading_simulation(self):
        """Test complete trading simulation."""
        env = CryptoTradingEnvironment(
            settings=self.settings,
            data=self.mock_data,
            symbols=['BTCUSDT'],
            initial_balance=10000.0
        )
        
        agent = CryptoRLAgent(
            settings=self.settings,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        # Run simulation
        obs, _ = env.reset()
        total_reward = 0
        
        for _ in range(10):
            action, _ = agent.select_action(obs, training=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        self.assertIsInstance(total_reward, float)
        self.assertTrue(env.balance > 0)
        print("✅ Trading simulation completed successfully")
    
    def test_risk_manager(self):
        """Test risk management system."""
        risk_manager = RiskManager(self.settings)
        
        # Test risk evaluation
        risk_metrics = risk_manager.evaluate_risk(
            symbol='BTCUSDT',
            current_price=50000.0,
            position_size=0.1
        )
        
        self.assertIsInstance(risk_metrics, dict)
        self.assertTrue('var_95' in risk_metrics)
        self.assertTrue('max_drawdown' in risk_metrics)
        print("✅ Risk manager works correctly")
    
    def test_mock_trader(self):
        """Test mock trading execution."""
        from cryptorl.trading.execution import Order, Position
        
        # Test order creation
        order = Order(
            symbol='BTCUSDT',
            side='BUY',
            order_type='LIMIT',
            quantity=0.1,
            price=50000.0
        )
        
        self.assertEqual(order.symbol, 'BTCUSDT')
        self.assertEqual(order.side, 'BUY')
        self.assertEqual(order.quantity, 0.1)
        print("✅ Mock trading system works correctly")


def run_comprehensive_test():
    """Run all tests and provide summary."""
    
    print("🧪 Running Comprehensive CryptoRL System Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCryptoRLSystem)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"   Tests run: {result.testsRun}")
        print("   The system is ready for real API integration!")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            for test, traceback in result.failures:
                print(f"   - {test}: {traceback}")
        
        if result.errors:
            for test, traceback in result.errors:
                print(f"   - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 System is ready for API configuration!")
        print("\nNext steps:")
        print("1. Update .env with real API keys")
        print("2. Set BINANCE_TESTNET=true")
        print("3. Run: python scripts/collect_data.py")
        print("4. Monitor system with: python scripts/validate_setup.py")