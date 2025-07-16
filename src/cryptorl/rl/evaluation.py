"""Evaluation utilities for CryptoRL agent."""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import gymnasium as gym
from datetime import datetime
import json

from .agent import CryptoRLAgent
from .environment import CryptoTradingEnvironment
from ..config.settings import Settings


class Evaluator:
    """Comprehensive evaluation suite for CryptoRL agent."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.results = {}
        
    def evaluate_agent(
        self,
        agent: CryptoRLAgent,
        test_data: pd.DataFrame,
        num_episodes: int = 100,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive evaluation across multiple metrics."""
        
        logger.info(f"Starting comprehensive evaluation for {num_episodes} episodes...")
        
        # Create test environment
        test_env = CryptoTradingEnvironment(
            settings=self.settings,
            data=test_data,
            symbols=test_data['symbol'].unique().tolist(),
            initial_balance=self.settings.rl.initial_balance,
            max_position_size=self.settings.rl.max_position_size,
            trading_fee=self.settings.rl.trading_fee,
            max_episode_length=self.settings.rl.max_episode_length
        )
        
        # Run evaluation
        evaluation_results = self._run_evaluation(agent, test_env, num_episodes, deterministic)
        
        # Calculate metrics
        metrics = self._calculate_metrics(evaluation_results)
        
        # Risk analysis
        risk_metrics = self._calculate_risk_metrics(evaluation_results)
        
        # Performance analysis
        performance_metrics = self._calculate_performance_metrics(evaluation_results)
        
        # Combine all results
        comprehensive_results = {
            'basic_metrics': metrics,
            'risk_metrics': risk_metrics,
            'performance_metrics': performance_metrics,
            'raw_results': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results = comprehensive_results
        return comprehensive_results
    
    def _run_evaluation(
        self,
        agent: CryptoRLAgent,
        env: CryptoTradingEnvironment,
        num_episodes: int,
        deterministic: bool
    ) -> List[Dict[str, Any]]:
        """Run evaluation episodes."""
        
        results = []
        
        for episode in range(num_episodes):
            observation, info = env.reset()
            episode_data = {
                'episode': episode,
                'symbol': info['symbol'],
                'steps': [],
                'trades': [],
                'final_value': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0
            }
            
            # Track portfolio values for drawdown calculation
            portfolio_values = [info.get('portfolio_value', self.settings.rl.initial_balance)]
            
            while True:
                # Select action
                action, _ = agent.select_action(observation, training=False)
                
                # Execute action
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                # Track data
                step_data = {
                    'step': len(episode_data['steps']),
                    'price': info['current_price'],
                    'action': float(action),
                    'position': info['position'],
                    'balance': info['balance'],
                    'portfolio_value': info['portfolio_value'],
                    'reward': reward
                }
                episode_data['steps'].append(step_data)
                
                portfolio_values.append(info['portfolio_value'])
                
                observation = next_observation
                
                if terminated or truncated:
                    break
            
            # Calculate episode metrics
            initial_value = self.settings.rl.initial_balance
            final_value = portfolio_values[-1]
            
            episode_data.update({
                'final_value': final_value,
                'total_return': (final_value - initial_value) / initial_value,
                'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                'portfolio_values': portfolio_values,
                'episode_length': len(episode_data['steps'])
            })
            
            results.append(episode_data)
        
        return results
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate basic evaluation metrics."""
        
        returns = [r['total_return'] for r in results]
        final_values = [r['final_value'] for r in results]
        episode_lengths = [r['episode_length'] for r in results]
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'median_return': np.median(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'win_rate': np.mean([r > 0 for r in returns]),
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'mean_episode_length': np.mean(episode_lengths),
            'total_episodes': len(results)
        }
    
    def _calculate_risk_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        
        returns = [r['total_return'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        
        # VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR (Conditional Value at Risk)
        cvar_95 = np.mean([r for r in returns if r <= var_95])
        cvar_99 = np.mean([r for r in returns if r <= var_99])
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
        
        # Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns else 0
        sortino_ratio = np.mean(returns) / (downside_std + 1e-8)
        
        # Calmar ratio
        max_drawdown = max(max_drawdowns) if max_drawdowns else 0
        calmar_ratio = np.mean(returns) / (max_drawdown + 1e-8)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'mean_max_drawdown': np.mean(max_drawdowns),
            'std_max_drawdown': np.std(max_drawdowns)
        }
    
    def _calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate advanced performance metrics."""
        
        # Calculate cumulative returns
        all_portfolio_values = []
        for result in results:
            all_portfolio_values.extend(result['portfolio_values'])
        
        # Calculate rolling metrics
        if len(all_portfolio_values) > 1:
            returns = np.diff(all_portfolio_values) / all_portfolio_values[:-1]
            
            # Hit rate (percentage of positive returns)
            hit_rate = np.mean([r > 0 for r in returns])
            
            # Average win/loss ratio
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
            
            # Profit factor
            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Annualized return (assuming daily data)
            total_return = (all_portfolio_values[-1] - all_portfolio_values[0]) / all_portfolio_values[0]
            annualized_return = total_return * (252 / len(all_portfolio_values)) if len(all_portfolio_values) > 0 else 0
            
            # Annualized volatility
            annualized_volatility = np.std(returns) * np.sqrt(252)
            
            return {
                'hit_rate': hit_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'profit_factor': profit_factor,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'annualized_sharpe': annualized_return / (annualized_volatility + 1e-8)
            }
        
        return {}
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values."""
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        
        if not self.results:
            return "No evaluation results available."
        
        report = f"""# CryptoRL Evaluation Report

Generated: {self.results['timestamp']}

## Basic Metrics
- Mean Return: {self.results['basic_metrics']['mean_return']:.4f} ({self.results['basic_metrics']['std_return']:.4f})
- Win Rate: {self.results['basic_metrics']['win_rate']:.2%}
- Mean Final Value: ${self.results['basic_metrics']['mean_final_value']:.2f}
- Total Episodes: {self.results['basic_metrics']['total_episodes']}

## Risk Metrics
- Sharpe Ratio: {self.results['risk_metrics']['sharpe_ratio']:.4f}
- Sortino Ratio: {self.results['risk_metrics']['sortino_ratio']:.4f}
- Max Drawdown: {self.results['risk_metrics']['max_drawdown']:.2%}
- Value at Risk (95%): {self.results['risk_metrics']['var_95']:.2%}
- Conditional VaR (95%): {self.results['risk_metrics']['cvar_95']:.2%}

## Performance Metrics
- Hit Rate: {self.results['performance_metrics'].get('hit_rate', 0):.2%}
- Win/Loss Ratio: {self.results['performance_metrics'].get('win_loss_ratio', 0):.4f}
- Profit Factor: {self.results['performance_metrics'].get('profit_factor', 0):.4f}
- Annualized Return: {self.results['performance_metrics'].get('annualized_return', 0):.2%}
- Annualized Sharpe: {self.results['performance_metrics'].get('annualized_sharpe', 0):.4f}

## Symbol Performance
"""
        
        # Add symbol-specific performance
        symbol_returns = {}
        for result in self.results['raw_results']:
            symbol = result['symbol']
            if symbol not in symbol_returns:
                symbol_returns[symbol] = []
            symbol_returns[symbol].append(result['total_return'])
        
        for symbol, returns in symbol_returns.items():
            mean_return = np.mean(returns)
            win_rate = np.mean([r > 0 for r in returns])
            report += f"- {symbol}: Mean={mean_return:.4f}, Win Rate={win_rate:.2%}\n"
        
        return report
    
    def save_results(self, filepath: str = "evaluation_results.json"):
        """Save evaluation results to file."""
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Evaluation results loaded from {filepath}")
        return self.results
    
    def compare_models(
        self,
        agents: List[Tuple[str, CryptoRLAgent]],
        test_data: pd.DataFrame,
        num_episodes: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple agents."""
        
        comparison_results = {}
        
        for name, agent in agents:
            logger.info(f"Evaluating {name}...")
            results = self.evaluate_agent(agent, test_data, num_episodes)
            comparison_results[name] = results
        
        # Create comparison summary
        summary = {}
        for metric in ['mean_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
            values = [(name, results['basic_metrics']['mean_return']) 
                     for name, results in comparison_results.items()]
            values.sort(key=lambda x: x[1], reverse=True)
            summary[metric] = values
        
        return comparison_results, summary


class BacktestEngine:
    """Backtesting engine for CryptoRL strategies."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.evaluator = Evaluator(settings)
    
    def run_backtest(
        self,
        agent: CryptoRLAgent,
        historical_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Run backtest on historical data."""
        
        # Filter data by date range
        mask = (historical_data['date'] >= start_date) & (historical_data['date'] <= end_date)
        backtest_data = historical_data[mask]
        
        if backtest_data.empty:
            logger.error("No data available for backtest period")
            return {}
        
        # Run evaluation
        results = self.evaluator.evaluate_agent(
            agent, 
            backtest_data, 
            num_episodes=1  # Single continuous backtest
        )
        
        # Add backtest-specific metrics
        results['backtest_period'] = {
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': len(backtest_data)
        }
        
        return results