#!/usr/bin/env python3
"""
Multi-Symbol CryptoRL Training Pipeline

Train a Mamba-based PPO agent on BTC, ETH, SOL using real data and LLM confidence vectors.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptorl.rl.environment import CryptoTradingEnvironment
from cryptorl.rl.agent import CryptoRLAgent
from cryptorl.rl.training import Trainer as PPOTrainer
from cryptorl.config.settings import Settings
from loguru import logger

# Remove get_logger call, use loguru directly

class MultiSymbolTrainer:
    """Multi-symbol training pipeline with real data."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.data_dir = Path("data/training")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.window_size = 24  # 24 hours
        self.initial_balance = 10000
        self.max_position_size = 0.1
        self.transaction_cost = 0.001
        
    def load_training_data(self) -> Dict[str, pd.DataFrame]:
        """Load prepared training data for all symbols."""
        logger.info("📊 Loading training data...")
        
        training_data = {}
        
        for symbol in self.symbols:
            train_path = self.data_dir / "splits" / f"{symbol}_train.csv"
            
            if not train_path.exists():
                logger.warning(f"⚠️ No training data for {symbol}, generating mock data...")
                training_data[symbol] = self._generate_mock_data(symbol)
            else:
                data = pd.read_csv(train_path, index_col=0, parse_dates=True)
                training_data[symbol] = data
                logger.info(f"✅ {symbol}: {len(data)} training records")
        
        return training_data
    
    def _generate_mock_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock training data if real data not available."""
        logger.info(f"🎲 Generating mock data for {symbol}...")
        
        # Generate 90 days of hourly data
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), 
                            periods=90*24, freq='H')
        
        np.random.seed(42)
        
        # Base price based on symbol
        base_prices = {'BTCUSDT': 40000, 'ETHUSDT': 2500, 'SOLUSDT': 100}
        base_price = base_prices.get(symbol, 1000)
        
        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, len(dates)),
            'rsi': np.random.uniform(20, 80, len(dates)),
            'macd': np.random.uniform(-100, 100, len(dates)),
            'fundamental_confidence': np.random.uniform(0.4, 0.9, len(dates)),
            'market_sentiment': np.random.uniform(0.3, 0.8, len(dates)),
            'regulatory_impact': np.random.uniform(0.2, 0.7, len(dates)),
            'tech_innovation': np.random.uniform(0.4, 0.9, len(dates)),
            'geopolitical_risk': np.random.uniform(0.1, 0.6, len(dates))
        }, index=dates)
        
        return data
    
    def create_multi_symbol_env(self, training_data: Dict[str, pd.DataFrame]) -> CryptoTradingEnvironment:
        """Create multi-symbol training environment."""
        logger.info("🎯 Creating multi-symbol training environment...")
        
        # Combine all symbols into training sequences
        all_data = []
        for symbol, data in training_data.items():
            if data.empty:
                continue
                
            # Add symbol identifier
            data = data.copy()
            data['symbol_id'] = self.symbols.index(symbol)
            all_data.append(data)
        
        if not all_data:
            raise ValueError("No training data available")
        
        # Concatenate all data
        combined_data = pd.concat(all_data)
        combined_data = combined_data.sort_index()
        
        logger.info(f"✅ Combined dataset: {len(combined_data)} records")
        
        # Create environment
        env = CryptoTradingEnvironment(
            settings=self.settings,
            data=combined_data,
            symbols=self.symbols,
            initial_balance=self.initial_balance,
            max_position_size=self.max_position_size,
            trading_fee=self.transaction_cost,
            max_episode_length=self.window_size
        )
        
        return env
    
    def train_agent(self, env: CryptoTradingEnvironment, total_timesteps: int = 100000) -> CryptoRLAgent:
        """Train the RL agent."""
        logger.info("🚀 Starting training...")
        
        # Initialize agent
        agent = CryptoRLAgent(
            settings=self.settings,
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=3e-4,
            gamma=0.99,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5
        )
        
        # Initialize trainer
        trainer = PPOTrainer(
            settings=self.settings,
            data_engine=None,  # Not used in this simplified training
            output_dir=str(self.models_dir)
        )
        
        # Train
        logger.info(f"Training for {total_timesteps} timesteps...")
        environments = [env]  # Wrap single environment in list
        training_metrics = trainer.train_agent(
            agent=agent,
            environments=environments,
            num_episodes=total_timesteps // 1000,  # Convert timesteps to episodes
            eval_frequency=50,
            save_frequency=100
        )
        
        # Save training metrics
        metrics_df = pd.DataFrame([training_metrics])
        metrics_df.to_csv(self.models_dir / "training_metrics.csv", index=False)
        
        logger.info("✅ Training completed!")
        return agent
    
    def validate_agent(self, agent: CryptoRLAgent, validation_data: Dict[str, pd.DataFrame]) -> Dict:
        """Validate the trained agent."""
        logger.info("🔍 Validating agent...")
        
        validation_results = {}
        
        for symbol, data in validation_data.items():
            if data.empty:
                continue
                
            # Create validation environment
            val_env = CryptoTradingEnvironment(
                settings=self.settings,
                data=data,
                symbols=[symbol],
                initial_balance=self.initial_balance,
                max_position_size=self.max_position_size,
                trading_fee=self.transaction_cost,
                max_episode_length=self.window_size
            )
            
            # Run validation episode
            obs, info = val_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = agent.select_action(obs, training=False)
                obs, reward, terminated, truncated, info = val_env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            validation_results[symbol] = {
                'total_reward': total_reward,
                'final_balance': info.get('balance', self.initial_balance),
                'final_portfolio_value': info.get('portfolio_value', self.initial_balance)
            }
        
        return validation_results
    
    def save_model(self, agent: CryptoRLAgent, model_name: str = ""):
        """Save the trained model."""
        if not model_name:
            model_name = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = self.models_dir / f"{model_name}.pth"
        torch.save({
            'model_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'settings': self.settings,
            'symbols': self.symbols,
            'training_date': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"💾 Model saved to {model_path}")
        return model_path
    
    def run(self, total_timesteps: int = 100000) -> bool:
        """Run complete training pipeline."""
        logger.info("🎯 Starting Multi-Symbol Training Pipeline")
        
        try:
            # Step 1: Load training data
            training_data = self.load_training_data()
            
            # Step 2: Load validation data
            val_data = {}
            for symbol in self.symbols:
                val_path = self.data_dir / "splits" / f"{symbol}_val.csv"
                if val_path.exists():
                    val_data[symbol] = pd.read_csv(val_path, index_col=0, parse_dates=True)
                else:
                    val_data[symbol] = self._generate_mock_data(symbol)
            
            # Step 3: Create environment
            env = self.create_multi_symbol_env(training_data)
            
            # Step 4: Train agent
            agent = self.train_agent(env, total_timesteps)
            
            # Step 5: Validate agent
            validation_results = self.validate_agent(agent, val_data)
            
            # Step 6: Save model
            model_path = self.save_model(agent)
            
            # Print summary
            print("\n" + "="*60)
            print("🎯 TRAINING SUMMARY")
            print("="*60)
            print(f"📊 Symbols trained: {', '.join(self.symbols)}")
            print(f"🎯 Total timesteps: {total_timesteps:,}")
            print(f"💾 Model saved: {model_path}")
            
            print("\n📈 Validation Results:")
            for symbol, results in validation_results.items():
                print(f"   {symbol}:")
                print(f"      Final Value: ${results['final_portfolio_value']:,.2f}")
                print(f"      Total Reward: {results['total_reward']:.2f}")
            
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train CryptoRL agent")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--symbols", nargs="+", default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize settings
    settings = Settings()
    
    # Create trainer
    trainer = MultiSymbolTrainer(settings)
    trainer.symbols = args.symbols
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Run training
    success = trainer.run(args.timesteps)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("Next step: python backtest.py")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()