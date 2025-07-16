"""Custom Gym environment for crypto trading with LLM-enhanced observations."""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import torch
from loguru import logger

from ..data.fusion import DataFusionEngine
from ..config.settings import Settings


class CryptoTradingEnvironment(gym.Env):
    """Custom environment for cryptocurrency trading with LLM confidence vectors."""
    
    def __init__(
        self,
        settings: Settings,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        max_position_size: float = 1.0,
        trading_fee: float = 0.001,
        max_episode_length: int = 30,
        symbols: List[str] = None
    ):
        super().__init__()
        
        self.settings = settings
        self.data = data
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.trading_fee = trading_fee
        self.max_episode_length = max_episode_length
        self.symbols = symbols or data['symbol'].unique().tolist()
        
        # Environment state
        self.current_step = 0
        self.episode_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Position size (-1 to 1)
        self.entry_price = 0.0
        self.current_symbol_idx = 0
        
        # Data management
        self.data_by_symbol = {symbol: data[data['symbol'] == symbol].copy() for symbol in self.symbols}
        self.current_data = None
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Calculate observation space dimensions
        self.feature_columns = self._get_feature_columns()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_columns),),
            dtype=np.float32
        )
        
        # Episode tracking
        self.episode_history = []
        self.trades = []
        
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns for observations."""
        
        # Include market features, confidence vectors, and account state
        market_features = [
            'close_norm', 'volume_norm', 'price_change',
            'rsi_norm', 'macd_norm', 'volume_ratio_norm'
        ]
        
        confidence_features = [
            'confidence_fundamentals', 'confidence_industry',
            'confidence_geopolitics', 'confidence_macro',
            'confidence_technical', 'confidence_regulatory',
            'confidence_innovation', 'confidence_overall'
        ]
        
        account_features = [
            'normalized_balance', 'position_size', 'unrealized_pnl',
            'portfolio_value'
        ]
        
        return market_features + confidence_features + account_features
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.episode_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        
        # Randomly select starting symbol and date
        if options and 'symbol' in options:
            self.current_symbol = options['symbol']
        else:
            self.current_symbol = np.random.choice(self.symbols)
        
        self.current_data = self.data_by_symbol[self.current_symbol]
        
        # Random starting point
        max_start_idx = len(self.current_data) - self.max_episode_length - 1
        if max_start_idx > 0:
            start_idx = np.random.randint(0, max_start_idx)
        else:
            start_idx = 0
        
        self.current_step = start_idx
        
        # Reset tracking
        self.episode_history = []
        self.trades = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        
        # Parse action
        target_position = float(np.clip(action[0], -1.0, 1.0))
        
        # Get current market data
        current_data = self.current_data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Execute trade
        reward, trade_executed = self._execute_trade(target_position, current_price)
        
        # Advance step
        self.current_step += 1
        self.episode_step += 1
        
        # Check termination
        terminated = (
            self.episode_step >= self.max_episode_length or
            self.current_step >= len(self.current_data) - 1 or
            self.balance <= 0
        )
        
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Track episode
        self.episode_history.append({
            'step': self.episode_step,
            'symbol': self.current_symbol,
            'price': current_price,
            'position': self.position,
            'balance': self.balance,
            'reward': reward,
            'portfolio_value': self._get_portfolio_value(current_price)
        })
        
        return observation, reward, terminated, truncated, info
    
    def _execute_trade(self, target_position: float, current_price: float) -> Tuple[float, bool]:
        """Execute a trade based on target position."""
        
        if abs(target_position - self.position) < 0.01:  # No significant change
            return 0.0, False
        
        # Calculate trade size
        position_change = target_position - self.position
        trade_size = abs(position_change) * self.balance
        
        # Calculate fees
        trade_fee = trade_size * self.trading_fee
        
        # Update balance and position
        self.balance -= trade_fee
        
        if position_change > 0:  # Buying
            if self.position >= 0:  # Long position
                self.balance -= trade_size
            else:  # Covering short
                self.balance += trade_size * (1 - (current_price - self.entry_price) / self.entry_price)
        else:  # Selling
            if self.position <= 0:  # Short position
                self.balance += trade_size
            else:  # Selling long
                self.balance += trade_size * (1 + (current_price - self.entry_price) / self.entry_price)
        
        # Update position and entry price
        self.position = target_position
        self.entry_price = current_price
        
        # Calculate reward (change in portfolio value)
        old_portfolio_value = self._get_portfolio_value(current_price)
        new_portfolio_value = self._get_portfolio_value(current_price)
        reward = new_portfolio_value - old_portfolio_value
        
        # Track trade
        self.trades.append({
            'step': self.episode_step,
            'symbol': self.current_symbol,
            'price': current_price,
            'position_change': position_change,
            'trade_size': trade_size,
            'fee': trade_fee,
            'reward': reward
        })
        
        return reward, True
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        
        if self.current_step >= len(self.current_data):
            return np.zeros(self.observation_space.shape)
        
        current_data = self.current_data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Market features
        market_features = []
        for col in ['close_norm', 'volume_norm', 'price_change', 'rsi_norm', 'macd_norm', 'volume_ratio_norm']:
            market_features.append(float(current_data.get(col, 0.0)))
        
        # Confidence features
        confidence_features = []
        for col in [
            'confidence_fundamentals', 'confidence_industry', 'confidence_geopolitics',
            'confidence_macro', 'confidence_technical', 'confidence_regulatory',
            'confidence_innovation', 'confidence_overall'
        ]:
            confidence_features.append(float(current_data.get(col, 0.5)))
        
        # Account state features
        portfolio_value = self._get_portfolio_value(current_price)
        unrealized_pnl = 0.0
        if self.position != 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price
        
        account_features = [
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            unrealized_pnl,  # Unrealized P&L
            portfolio_value / self.initial_balance  # Normalized portfolio value
        ]
        
        observation = np.array(market_features + confidence_features + account_features, dtype=np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for debugging."""
        
        if self.current_step >= len(self.current_data):
            return {}
        
        current_data = self.current_data.iloc[self.current_step]
        current_price = current_data['close']
        
        return {
            'symbol': self.current_symbol,
            'current_price': current_price,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self._get_portfolio_value(current_price),
            'step': self.episode_step,
            'total_steps': len(self.current_data),
            'date': str(current_data.get('date', ''))
        }
    
    def _get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        
        position_value = self.position * self.balance * current_price
        return self.balance + position_value
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the completed episode."""
        
        if not self.episode_history:
            return {}
        
        initial_value = self.initial_balance
        final_value = self.episode_history[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate metrics
        values = [h['portfolio_value'] for h in self.episode_history]
        max_value = max(values)
        min_value = min(values)
        max_drawdown = (max_value - min_value) / max_value if max_value > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = [h['reward'] for h in self.episode_history if h['reward'] != 0]
        if returns:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            sharpe_ratio = 0.0
        
        return {
            'symbol': self.current_symbol,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'win_rate': sum(1 for t in self.trades if t['reward'] > 0) / max(len(self.trades), 1),
            'episode_length': len(self.episode_history)
        }


class MultiSymbolTradingEnvironment(gym.Env):
    """Environment that handles multiple symbols sequentially."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.kwargs = kwargs
        self.current_env_idx = 0
        self.environments = []
        
        # Create environments for each symbol
        symbols = kwargs.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        for symbol in symbols:
            env_kwargs = kwargs.copy()
            env_kwargs['symbols'] = [symbol]
            self.environments.append(CryptoTradingEnvironment(**env_kwargs))
    
    def reset(self, **kwargs):
        """Reset and return to first environment."""
        self.current_env_idx = 0
        return self.environments[0].reset(**kwargs)
    
    def step(self, action):
        """Step through current environment."""
        obs, reward, terminated, truncated, info = self.environments[self.current_env_idx].step(action)
        
        # Move to next environment when current one ends
        if terminated or truncated:
            self.current_env_idx = (self.current_env_idx + 1) % len(self.environments)
            if self.current_env_idx == 0:  # Completed all symbols
                return obs, reward, True, truncated, info
        
        return obs, reward, False, truncated, info
    
    @property
    def observation_space(self):
        return self.environments[0].observation_space
    
    @property
    def action_space(self):
        return self.environments[0].action_space