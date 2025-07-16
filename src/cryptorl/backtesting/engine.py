"""Comprehensive backtesting engine for CryptoRL strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..rl.environment import CryptoTradingEnvironment
from ..rl.agent import CryptoRLAgent
from ..config.settings import Settings


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk metrics
    var_95: float
    var_99: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    
    # Timing metrics
    avg_holding_period: float
    max_holding_period: float
    min_holding_period: float
    
    # Portfolio metrics
    final_value: float
    peak_value: float
    final_balance: float
    
    # Market metrics
    correlation_with_market: float
    beta: float
    alpha: float
    
    # Drawdown periods
    drawdown_periods: List[Tuple[datetime, datetime, float]]
    
    # Trade log
    trade_log: pd.DataFrame
    portfolio_history: pd.DataFrame
    
    # Confidence vector analysis
    confidence_correlation: Dict[str, float]
    confidence_impact: Dict[str, float]


class BacktestingEngine:
    """High-precision backtesting engine for CryptoRL strategies."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def run_single_backtest(
        self,
        agent: CryptoRLAgent,
        test_data: pd.DataFrame,
        symbol: str,
        initial_balance: float = 10000.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """Run a single backtest on historical data."""
        
        self.logger.info(f"Running backtest for {symbol}")
        
        # Filter data by date range
        if start_date:
            test_data = test_data[test_data['date'] >= start_date]
        if end_date:
            test_data = test_data[test_data['date'] <= end_date]
            
        # Create environment
        env = CryptoTradingEnvironment(
            settings=self.settings,
            data=test_data,
            symbols=[symbol],
            initial_balance=initial_balance,
            max_position_size=self.settings.rl_max_position_size,
            trading_fee=self.settings.rl_trading_fee,
            max_episode_length=len(test_data)
        )
        
        # Run episode
        observation, info = env.reset()
        done = False
        
        # Track metrics
        portfolio_values = [info.get('portfolio_value', initial_balance)]
        balances = [info.get('balance', initial_balance)]
        positions = [info.get('position', 0)]
        timestamps = [test_data.iloc[0]['date']]
        
        # Trade tracking
        trades = []
        position_start = None
        position_value = 0
        
        while not done:
            # Get action from agent
            action, _ = agent.select_action(observation, training=False)
            
            # Store current position state
            prev_position = positions[-1]
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track portfolio metrics
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
            balances.append(info.get('balance', balances[-1]))
            positions.append(info.get('position', positions[-1]))
            timestamps.append(test_data.iloc[len(portfolio_values)-1]['date'])
            
            # Track trades
            if prev_position != positions[-1] or done:
                if prev_position != 0:  # Closing position
                    if position_start is not None:
                        trade_return = (portfolio_values[-1] - position_value) / position_value
                        trades.append({
                            'entry_time': position_start,
                            'exit_time': timestamps[-1],
                            'entry_price': position_value,
                            'exit_price': portfolio_values[-1],
                            'return': trade_return,
                            'duration': (timestamps[-1] - position_start).days,
                            'position_type': 'long' if prev_position > 0 else 'short'
                        })
                        position_start = None
                        position_value = 0
                
                if positions[-1] != 0:  # Opening new position
                    position_start = timestamps[-1]
                    position_value = portfolio_values[-1]
        
        # Convert to DataFrame
        trade_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame({
            'timestamp': timestamps,
            'portfolio_value': portfolio_values,
            'balance': balances,
            'position': positions
        })
        
        # Calculate metrics
        result = self._calculate_metrics(
            trade_df, portfolio_df, test_data, symbol
        )
        
        return result
    
    def run_walk_forward_analysis(
        self,
        agent: CryptoRLAgent,
        data: pd.DataFrame,
        symbol: str,
        train_period: int = 252,  # 1 year
        test_period: int = 63,    # 3 months
        step_size: int = 21       # 1 month
    ) -> List[BacktestResult]:
        """Run walk-forward analysis."""
        
        self.logger.info(f"Running walk-forward analysis for {symbol}")
        
        results = []
        data = data.sort_values('date')
        
        start_idx = 0
        while start_idx + train_period + test_period <= len(data):
            train_end = start_idx + train_period
            test_end = train_end + test_period
            
            # Training period
            train_data = data.iloc[start_idx:train_end]
            
            # Test period
            test_data = data.iloc[train_end:test_end]
            
            # Create training environment and retrain agent
            train_env = CryptoTradingEnvironment(
                settings=self.settings,
                data=train_data,
                symbols=[symbol],
                initial_balance=self.settings.rl_initial_balance,
                max_position_size=self.settings.rl_max_position_size,
                trading_fee=self.settings.rl_trading_fee,
                max_episode_length=len(train_data)
            )
            
            # Quick retraining for walk-forward
            for episode in range(5):  # Quick retraining
                agent.train_episode(train_env)
            
            # Run backtest on test period
            result = self.run_single_backtest(agent, test_data, symbol)
            results.append(result)
            
            # Move window forward
            start_idx += step_size
        
        return results
    
    def _calculate_metrics(
        self,
        trades: pd.DataFrame,
        portfolio_df: pd.DataFrame,
        market_data: pd.DataFrame,
        symbol: str
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        
        if trades.empty:
            return self._empty_result(portfolio_df)
        
        # Basic return metrics
        portfolio_values = portfolio_df['portfolio_value'].values
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized return
        days = (portfolio_df['timestamp'].iloc[-1] - portfolio_df['timestamp'].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # Volatility and Sharpe ratio
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak_values = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - peak_values) / peak_values
        max_drawdown = np.min(drawdowns)
        
        # VaR calculations
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        var_99 = np.percentile(daily_returns, 1) if len(daily_returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade metrics
        total_trades = len(trades)
        win_rate = len(trades[trades['return'] > 0]) / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades[trades['return'] > 0]['return'].sum()
        gross_loss = abs(trades[trades['return'] < 0]['return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_trade_return = trades['return'].mean()
        avg_win = trades[trades['return'] > 0]['return'].mean() if win_rate > 0 else 0
        avg_loss = trades[trades['return'] < 0]['return'].mean() if win_rate < 1 else 0
        
        # Holding periods
        avg_holding_period = trades['duration'].mean()
        max_holding_period = trades['duration'].max()
        min_holding_period = trades['duration'].min()
        
        # Market correlation
        market_returns = market_data['close'].pct_change().dropna().values
        if len(market_returns) > 1 and len(daily_returns) > 1:
            correlation = np.corrcoef(market_returns, daily_returns[:len(market_returns)])[0, 1]
            correlation = 0 if np.isnan(correlation) else correlation
        else:
            correlation = 0
        
        # Beta and Alpha
        market_return = (market_data['close'].iloc[-1] - market_data['close'].iloc[0]) / market_data['close'].iloc[0]
        market_annualized = (1 + market_return) ** (365 / max(days, 1)) - 1
        beta = correlation * (volatility / (np.std(market_returns) * np.sqrt(252))) if len(market_returns) > 0 else 0
        alpha = annualized_return - (0.02 + beta * (market_annualized - 0.02))
        
        # Drawdown periods
        drawdown_periods = self._identify_drawdown_periods(portfolio_df)
        
        # Confidence vector analysis
        confidence_correlation, confidence_impact = self._analyze_confidence_vectors(
            trades, market_data
        )
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_period=avg_holding_period,
            max_holding_period=max_holding_period,
            min_holding_period=min_holding_period,
            final_value=portfolio_values[-1],
            peak_value=np.max(portfolio_values),
            final_balance=portfolio_df['balance'].iloc[-1],
            correlation_with_market=correlation,
            beta=beta,
            alpha=alpha,
            drawdown_periods=drawdown_periods,
            trade_log=trades,
            portfolio_history=portfolio_df,
            confidence_correlation=confidence_correlation,
            confidence_impact=confidence_impact
        )
    
    def _empty_result(self, portfolio_df: pd.DataFrame) -> BacktestResult:
        """Return empty result when no trades."""
        
        portfolio_values = portfolio_df['portfolio_value'].values
        return BacktestResult(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            var_99=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_return=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_holding_period=0.0,
            max_holding_period=0.0,
            min_holding_period=0.0,
            final_value=portfolio_values[-1],
            peak_value=np.max(portfolio_values),
            final_balance=portfolio_df['balance'].iloc[-1],
            correlation_with_market=0.0,
            beta=0.0,
            alpha=0.0,
            drawdown_periods=[],
            trade_log=pd.DataFrame(),
            portfolio_history=portfolio_df,
            confidence_correlation={},
            confidence_impact={}
        )
    
    def _identify_drawdown_periods(self, portfolio_df: pd.DataFrame) -> List[Tuple[datetime, datetime, float]]:
        """Identify drawdown periods."""
        
        drawdown_periods = []
        portfolio_values = portfolio_df['portfolio_value'].values
        
        peak_values = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - peak_values) / peak_values
        
        # Find drawdown periods
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdowns):
            if dd < -0.05 and not in_drawdown:  # 5% threshold
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                start_date = portfolio_df['timestamp'].iloc[start_idx]
                end_date = portfolio_df['timestamp'].iloc[i]
                max_dd = np.min(drawdowns[start_idx:i+1])
                drawdown_periods.append((start_date, end_date, max_dd))
        
        return drawdown_periods
    
    def _analyze_confidence_vectors(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Analyze correlation between confidence vectors and returns."""
        
        confidence_features = [
            'confidence_fundamentals',
            'confidence_industry',
            'confidence_geopolitics',
            'confidence_macro',
            'confidence_technical',
            'confidence_regulatory',
            'confidence_innovation',
            'confidence_overall'
        ]
        
        correlations = {}
        impacts = {}
        
        for feature in confidence_features:
            if feature in market_data.columns and len(trades) > 0:
                # Match trades with confidence vectors
                trade_confidences = []
                trade_returns = []
                
                for _, trade in trades.iterrows():
                    # Find closest market data point
                    closest_idx = (market_data['date'] - trade['entry_time']).abs().idxmin()
                    if abs((market_data.loc[closest_idx, 'date'] - trade['entry_time']).days) <= 1:
                        trade_confidences.append(market_data.loc[closest_idx, feature])
                        trade_returns.append(trade['return'])
                
                if trade_confidences and trade_returns:
                    correlation = np.corrcoef(trade_confidences, trade_returns)[0, 1]
                    correlations[feature] = 0 if np.isnan(correlation) else correlation
                    
                    # Calculate impact: average return when confidence > 0.7 vs < 0.3
                    high_conf = [r for c, r in zip(trade_confidences, trade_returns) if c > 0.7]
                    low_conf = [r for c, r in zip(trade_confidences, trade_returns) if c < 0.3]
                    
                    if high_conf and low_conf:
                        impacts[feature] = np.mean(high_conf) - np.mean(low_conf)
                    else:
                        impacts[feature] = 0.0
                else:
                    correlations[feature] = 0.0
                    impacts[feature] = 0.0
            else:
                correlations[feature] = 0.0
                impacts[feature] = 0.0
        
        return correlations, impacts
    
    def generate_report(
        self,
        results: Dict[str, List[BacktestResult]],
        output_dir: str = "./backtest_reports"
    ) -> str:
        """Generate comprehensive backtest report."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Aggregate results
        summary_stats = {}
        for symbol, symbol_results in results.items():
            if symbol_results:
                # Average metrics across all runs
                summary_stats[symbol] = {
                    'total_return': np.mean([r.total_return for r in symbol_results]),
                    'sharpe_ratio': np.mean([r.sharpe_ratio for r in symbol_results]),
                    'max_drawdown': np.mean([r.max_drawdown for r in symbol_results]),
                    'win_rate': np.mean([r.win_rate for r in symbol_results]),
                    'total_trades': np.mean([r.total_trades for r in symbol_results])
                }
        
        # Create report
        report = f"""# CryptoRL Backtest Report

## Summary
- **Symbols Tested**: {len(results)}
- **Total Backtests Run**: {sum(len(r) for r in results.values())}

## Performance Summary

| Symbol | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|--------|--------------|--------------|--------------|----------|--------|
"""
        
        for symbol, stats in summary_stats.items():
            report += f"| {symbol} | {stats['total_return']:.2%} | {stats['sharpe_ratio']:.2f} | {stats['max_drawdown']:.2%} | {stats['win_rate']:.2%} | {int(stats['total_trades'])} |\n"
        
        report += """
## Detailed Analysis

### Risk Metrics
- **Value at Risk (95%)**: Average across all symbols
- **Calmar Ratio**: Risk-adjusted performance
- **Sortino Ratio**: Downside risk focus

### Confidence Vector Impact
Analysis of how LLM-generated confidence vectors correlate with trading performance.

### Drawdown Analysis
Detailed breakdown of significant drawdown periods and their characteristics.
"""
        
        # Save report
        report_path = Path(output_dir) / "backtest_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save detailed results
        results_data = {}
        for symbol, symbol_results in results.items():
            results_data[symbol] = []
            for result in symbol_results:
                results_data[symbol].append({
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'final_value': result.final_value,
                    'var_95': result.var_95,
                    'var_99': result.var_99
                })
        
        with open(Path(output_dir) / "backtest_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        return str(report_path)