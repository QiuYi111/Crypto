"""
Reinforcement Learning Environment for Crypto Trading

This module implements the trading environment for RL agents using the enhanced dataset
with LLM confidence vectors as additional state features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from gym import Env, spaces
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CryptoTradingEnvironment(Env):
    """
    Multi-asset cryptocurrency trading environment with LLM-enhanced observations.
    
    Features:
    - Multi-asset support (BTC, ETH, SOL)
    - LLM confidence vectors as state features
    - Realistic transaction costs and slippage
    - Risk management constraints
    - Dynamic position sizing
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        max_position_size: float = 1.0,
        lookback_window: int = 50,
        confidence_features: List[str] = None
    ):
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.confidence_features = confidence_features or [
            'fundamentals', 'market_sentiment', 'regulatory_impact',
            'tech_innovation', 'geopolitical', 'market_structure', 'risk_factors'
        ]
        
        # Extract unique assets
        self.assets = sorted(self.data['symbol'].unique())
        self.num_assets = len(self.assets)
        
        # State dimensions
        self.price_features = 4 * self.num_assets  # OHLC for each asset
        self.volume_features = self.num_assets
        self.technical_features = 8 * self.num_assets  # RSI, MACD, Bollinger, etc.
        self.confidence_dim = len(self.confidence_features)
        self.account_features = 3  # balance, position, unrealized_pnl
        
        self.state_dim = (
            self.price_features + 
            self.volume_features + 
            self.technical_features + 
            self.confidence_dim + 
            self.account_features
        )
        
        # Action space: [-1, 1] for each asset (-1 = short, 1 = long)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_assets,), 
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, self.state_dim),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = np.zeros(self.num_assets)
        self.entry_prices = np.zeros(self.num_assets)
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state."""
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        # Get data slice for all assets
        slice_data = self.data.iloc[start_idx:end_idx]
        
        # Initialize state array
        state = np.zeros((self.lookback_window, self.state_dim))
        
        for i, asset in enumerate(self.assets):
            asset_data = slice_data[slice_data['symbol'] == asset]
            
            if len(asset_data) < self.lookback_window:
                continue
                
            # Price features (OHLC)
            base_idx = i * 4
            state[:, base_idx:base_idx+4] = asset_data[['open', 'high', 'low', 'close']].values
            
            # Volume features
            vol_idx = self.price_features + i
            state[:, vol_idx] = asset_data['volume'].values
            
            # Technical indicators
            tech_start = self.price_features + self.volume_features + i * 8
            tech_cols = [
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                'atr', 'volume_ma', 'volatility'
            ]
            state[:, tech_start:tech_start+8] = asset_data[tech_cols].values
            
            # Confidence features (same for all assets in same time step)
            conf_start = self.price_features + self.volume_features + self.technical_features
            for j, conf_feat in enumerate(self.confidence_features):
                state[:, conf_start + j] = asset_data[conf_feat].values
        
        # Account features (last 3 columns)
        account_start = self.state_dim - 3
        state[:, account_start] = self.balance / self.initial_balance  # Normalized balance
        state[:, account_start + 1] = np.sum(np.abs(self.positions)) / self.max_position_size  # Position ratio
        state[:, account_start + 2] = self._get_unrealized_pnl() / self.initial_balance  # Normalized PnL
        
        return state.astype(np.float32)
    
    def _get_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL from open positions."""
        pnl = 0.0
        for i, asset in enumerate(self.assets):
            if self.positions[i] != 0:
                current_price = self._get_current_price(asset)
                pnl += self.positions[i] * (current_price - self.entry_prices[i])
        return pnl
    
    def _get_current_price(self, asset: str) -> float:
        """Get current price for an asset."""
        asset_data = self.data[
            (self.data['symbol'] == asset) & 
            (self.data.index == self.current_step)
        ]
        return asset_data['close'].iloc[0] if not asset_data.empty else 0.0
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        action = np.clip(action, -1, 1)
        
        # Calculate portfolio value before action
        old_portfolio_value = self.balance + self._get_unrealized_pnl()
        
        # Execute trades
        total_cost = 0.0
        for i, asset in enumerate(self.assets):
            current_price = self._get_current_price(asset)
            if current_price <= 0:
                continue
                
            # Calculate target position value
            target_value = action[i] * self.balance * self.max_position_size
            target_shares = target_value / current_price
            
            # Calculate trade amount
            current_shares = self.positions[i]
            trade_shares = target_shares - current_shares
            
            if abs(trade_shares) > 0.0001:  # Minimum trade threshold
                trade_cost = abs(trade_shares) * current_price
                fee = trade_cost * self.transaction_fee
                
                if self.balance >= trade_cost + fee:
                    self.balance -= trade_cost + fee
                    self.positions[i] = target_shares
                    
                    if target_shares > 0:
                        self.entry_prices[i] = current_price
                    
                    self.trade_history.append({
                        'step': self.current_step,
                        'asset': asset,
                        'action': 'buy' if trade_shares > 0 else 'sell',
                        'shares': abs(trade_shares),
                        'price': current_price,
                        'fee': fee
                    })
        
        # Calculate new portfolio value
        new_portfolio_value = self.balance + self._get_unrealized_pnl()
        
        # Calculate reward (PnL change)
        reward = new_portfolio_value - old_portfolio_value
        
        # Normalize reward
        reward = reward / self.initial_balance
        
        # Add risk penalty
        position_penalty = np.sum(np.abs(self.positions)) * 0.001
        reward -= position_penalty
        
        # Check if episode is done
        done = (
            self.current_step >= len(self.data) - 1 or
            new_portfolio_value <= self.initial_balance * 0.1  # 90% drawdown
        )
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        next_obs = self._get_observation() if not done else None
        
        info = {
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'unrealized_pnl': self._get_unrealized_pnl(),
            'total_return': (new_portfolio_value - self.initial_balance) / self.initial_balance,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
        
        return next_obs, reward, done, info
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade history."""
        if len(self.trade_history) < 2:
            return 0.0
        
        returns = [trade.get('return', 0) for trade in self.trade_history]
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def render(self, mode: str = 'human') -> None:
        """Render current state."""
        portfolio_value = self.balance + self._get_unrealized_pnl()
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance * 100
        
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Positions: {[f'{pos:.4f}' for pos in self.positions]}")
        print(f"Unrealized PnL: ${self._get_unrealized_pnl():.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print("-" * 50)


class BacktestingEngine:
    """
    Backtesting engine for evaluating RL strategies on historical data.
    """
    
    def __init__(self, environment: CryptoTradingEnvironment):
        self.env = environment
        self.results = []
    
    def run_backtest(self, model, num_episodes: int = 1) -> Dict[str, Any]:
        """Run backtest with given model."""
        all_results = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_results = []
            
            while not done:
                action = model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                episode_results.append(info)
            
            all_results.append(episode_results)
        
        return self._calculate_performance_metrics(all_results)
    
    def _calculate_performance_metrics(self, results: List[List[Dict]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        flat_results = [item for sublist in results for item in sublist]
        
        portfolio_values = [r['portfolio_value'] for r in flat_results]
        returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                  for i in range(1, len(portfolio_values))]
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_returns = [r for r in returns if r > 0]
        win_rate = len(positive_returns) / len(returns) if returns else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'portfolio_values': portfolio_values,
            'returns': returns,
            'final_balance': portfolio_values[-1]
        }