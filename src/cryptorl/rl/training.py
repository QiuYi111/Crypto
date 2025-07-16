"""Training utilities for CryptoRL agent."""

import os
import json
import time
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from loguru import logger
import pandas as pd
from tqdm import tqdm
import gymnasium as gym

from .agent import CryptoRLAgent
from .environment import CryptoTradingEnvironment
from ..data.fusion import DataFusionEngine
from ..config.settings import Settings


class Trainer:
    """Main training orchestrator for CryptoRL agent."""
    
    def __init__(
        self,
        settings: Settings,
        data_engine: DataFusionEngine,
        output_dir: str = "./models"
    ):
        self.settings = settings
        self.data_engine = data_engine
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training metrics
        self.training_history = []
        self.best_performance = float('-inf')
        
    async def prepare_training_data(self) -> pd.DataFrame:
        """Prepare enhanced training dataset."""
        
        logger.info("Preparing training data...")
        
        # Get training date range
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=self.settings.rl.training_days)
        
        # Create enhanced dataset
        symbols = self.settings.trading.symbols
        training_data = await self.data_engine.create_enhanced_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        logger.info(f"Training data prepared: {len(training_data)} rows across {len(symbols)} symbols")
        return training_data
    
    def create_training_environments(self, training_data: pd.DataFrame) -> List[CryptoTradingEnvironment]:
        """Create training environments."""
        
        environments = []
        
        # Create environments for each symbol
        for symbol in training_data['symbol'].unique():
            symbol_data = training_data[training_data['symbol'] == symbol]
            
            # Split data for training/validation
            train_size = int(0.8 * len(symbol_data))
            train_data = symbol_data[:train_size]
            val_data = symbol_data[train_size:]
            
            # Create training environment
            train_env = CryptoTradingEnvironment(
                settings=self.settings,
                data=train_data,
                symbols=[symbol],
                initial_balance=self.settings.rl.initial_balance,
                max_position_size=self.settings.rl.max_position_size,
                trading_fee=self.settings.rl.trading_fee,
                max_episode_length=self.settings.rl.max_episode_length
            )
            
            # Create validation environment
            val_env = CryptoTradingEnvironment(
                settings=self.settings,
                data=val_data,
                symbols=[symbol],
                initial_balance=self.settings.rl.initial_balance,
                max_position_size=self.settings.rl.max_position_size,
                trading_fee=self.settings.rl.trading_fee,
                max_episode_length=self.settings.rl.max_episode_length
            )
            
            environments.extend([train_env, val_env])
        
        return environments
    
    def train_agent(
        self,
        agent: CryptoRLAgent,
        environments: List[CryptoTradingEnvironment],
        num_episodes: int = 1000,
        eval_frequency: int = 50,
        save_frequency: int = 100
    ) -> Dict[str, Any]:
        """Train the RL agent."""
        
        logger.info(f"Starting training for {num_episodes} episodes...")
        
        training_metrics = {
            'episode_rewards': [],
            'training_losses': [],
            'validation_metrics': [],
            'training_time': 0
        }
        
        start_time = time.time()
        
        # Training loop
        for episode in tqdm(range(num_episodes), desc="Training"):
            # Select environment (rotate through available environments)
            env = environments[episode % len(environments)]
            
            # Train episode
            episode_metrics = agent.train_episode(env)
            training_metrics['episode_rewards'].append(episode_metrics['episode_reward'])
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward={episode_metrics['episode_reward']:.2f}, "
                          f"Final Value={episode_metrics['final_portfolio_value']:.2f}")
            
            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                eval_metrics = self.evaluate_agent(agent, environments)
                training_metrics['validation_metrics'].append({
                    'episode': episode,
                    **eval_metrics
                })
                
                # Save best model
                if eval_metrics['mean_reward'] > self.best_performance:
                    self.best_performance = eval_metrics['mean_reward']
                    self.save_checkpoint(agent, episode, "best")
                    logger.info(f"New best model saved at episode {episode}")
            
            # Save checkpoint
            if episode % save_frequency == 0:
                self.save_checkpoint(agent, episode, f"checkpoint_{episode}")
        
        training_time = time.time() - start_time
        training_metrics['training_time'] = training_time
        
        # Save final model
        self.save_checkpoint(agent, num_episodes, "final")
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return training_metrics
    
    def evaluate_agent(
        self,
        agent: CryptoRLAgent,
        environments: List[CryptoTradingEnvironment],
        num_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate agent performance across environments."""
        
        all_eval_metrics = []
        
        for env in environments[:min(len(environments), 3)]:  # Evaluate on first 3 environments
            eval_metrics = agent.evaluate(env, num_eval_episodes)
            all_eval_metrics.append(eval_metrics)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in ['mean_reward', 'std_reward', 'mean_final_value', 'win_rate']:
            values = [m[key] for m in all_eval_metrics]
            aggregated_metrics[key] = np.mean(values)
            aggregated_metrics[f'{key}_std'] = np.std(values)
        
        return aggregated_metrics
    
    def save_checkpoint(
        self,
        agent: CryptoRLAgent,
        episode: int,
        checkpoint_name: str
    ):
        """Save training checkpoint."""
        
        checkpoint_path = os.path.join(self.output_dir, f"{checkpoint_name}.pth")
        agent.save_model(checkpoint_path)
        
        # Save training history
        history_path = os.path.join(self.output_dir, f"{checkpoint_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                'episode': episode,
                'training_stats': agent.get_training_stats(),
                'settings': self.settings.dict()
            }, f, indent=2)
    
    def load_checkpoint(self, agent: CryptoRLAgent, checkpoint_name: str):
        """Load training checkpoint."""
        
        checkpoint_path = os.path.join(self.output_dir, f"{checkpoint_name}.pth")
        if os.path.exists(checkpoint_path):
            agent.load_model(checkpoint_path)
            logger.info(f"Loaded checkpoint: {checkpoint_name}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    def plot_training_progress(self, training_metrics: Dict[str, Any]) -> str:
        """Create training progress visualization."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Episode rewards
            axes[0, 0].plot(training_metrics['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            
            # Moving average rewards
            if len(training_metrics['episode_rewards']) >= 10:
                moving_avg = pd.Series(training_metrics['episode_rewards']).rolling(10).mean()
                axes[0, 1].plot(moving_avg)
                axes[0, 1].set_title('10-Episode Moving Average Reward')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Moving Average Reward')
            
            # Validation metrics
            if training_metrics['validation_metrics']:
                episodes = [m['episode'] for m in training_metrics['validation_metrics']]
                val_rewards = [m['mean_reward'] for m in training_metrics['validation_metrics']]
                axes[1, 0].plot(episodes, val_rewards, marker='o')
                axes[1, 0].set_title('Validation Rewards')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Mean Validation Reward')
            
            # Training time
            axes[1, 1].text(0.5, 0.5, f"Training Time: {training_metrics['training_time']:.2f}s\n"
                          f"Total Episodes: {len(training_metrics['episode_rewards'])}",
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, 'training_progress.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except ImportError:
            logger.warning("Matplotlib/seaborn not available for plotting")
            return None
    
    def generate_training_report(self, training_metrics: Dict[str, Any]) -> str:
        """Generate comprehensive training report."""
        
        report = f"""
# CryptoRL Training Report

## Training Summary
- Total Episodes: {len(training_metrics['episode_rewards'])}
- Training Time: {training_metrics['training_time']:.2f} seconds
- Best Performance: {self.best_performance:.4f}

## Performance Metrics
- Final Episode Reward: {training_metrics['episode_rewards'][-1]:.4f}
- Average Reward (Last 100): {np.mean(training_metrics['episode_rewards'][-100:]):.4f}
- Reward Std Dev: {np.std(training_metrics['episode_rewards']):.4f}

## Validation Results
"""
        
        if training_metrics['validation_metrics']:
            latest_validation = training_metrics['validation_metrics'][-1]
            report += f"""
- Mean Validation Reward: {latest_validation['mean_reward']:.4f}
- Validation Win Rate: {latest_validation['win_rate']:.4f}
- Mean Final Portfolio Value: {latest_validation['mean_final_value']:.4f}
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'training_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path


class HyperparameterOptimizer:
    """Hyperparameter optimization for RL training."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def grid_search(
        self,
        param_space: Dict[str, List],
        training_data: pd.DataFrame,
        num_trials: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform grid search over hyperparameters."""
        
        results = []
        
        # Generate parameter combinations
        import itertools
        keys = list(param_space.keys())
        values = list(param_space.values())
        
        combinations = list(itertools.product(*values))
        
        for i, combination in enumerate(combinations[:num_trials]):
            params = dict(zip(keys, combination))
            
            logger.info(f"Trial {i+1}/{len(combinations)}: {params}")
            
            # Update settings
            for key, value in params.items():
                if hasattr(self.settings.rl, key):
                    setattr(self.settings.rl, key, value)
            
            # Run training
            trainer = Trainer(self.settings, data_engine=None)
            # Note: This would need proper data_engine setup for full functionality
            
            results.append({
                'params': params,
                'performance': None  # Would contain actual results
            })
        
        return results