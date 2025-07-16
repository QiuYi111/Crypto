#!/usr/bin/env python3
"""
Comprehensive Multi-Symbol Backtesting Pipeline

Test the trained agent across BTC, ETH, SOL with detailed performance analysis,
walk-forward testing, and risk metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptorl.rl.environment import CryptoTradingEnvironment
from cryptorl.rl.agent import CryptoRLAgent
from cryptorl.backtesting.engine import BacktestingEngine, BacktestResult
from cryptorl.config.settings import Settings
from cryptorl.utils.logger import get_logger
from cryptorl.evaluation.metrics import calculate_all_metrics

logger = get_logger(__name__)

class ComprehensiveBacktestPipeline:
    """Comprehensive backtesting pipeline for multi-symbol evaluation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.data_dir = Path("data/training")
        self.models_dir = Path("models")
        self.results_dir = Path("results/backtests")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Backtesting parameters
        self.window_size = 24
        self.initial_balance = 10000
        self.max_position_size = 0.1
        self.transaction_cost = 0.001
        
        # Initialize backtesting engine
        self.backtest_engine = BacktestingEngine(settings)
    
    def load_test_data(self) -> Dict[str, pd.DataFrame]:
        """Load test data for all symbols."""
        logger.info("📊 Loading test data...")
        
        test_data = {}
        
        for symbol in self.symbols:
            test_path = self.data_dir / "splits" / f"{symbol}_test.csv"
            
            if test_path.exists():
                data = pd.read_csv(test_path, index_col=0, parse_dates=True)
                test_data[symbol] = data
                logger.info(f"✅ {symbol}: {len(data)} test records")
            else:
                logger.warning(f"⚠️ No test data for {symbol}, using validation data...")
                val_path = self.data_dir / "splits" / f"{symbol}_val.csv"
                if val_path.exists():
                    data = pd.read_csv(val_path, index_col=0, parse_dates=True)
                    test_data[symbol] = data
                else:
                    test_data[symbol] = self._generate_mock_test_data(symbol)
        
        return test_data
    
    def _generate_mock_test_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock test data if real data not available."""
        logger.info(f"🎲 Generating mock test data for {symbol}...")
        
        # Generate 30 days of hourly test data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            periods=30*24, freq='H')
        
        np.random.seed(123)  # Different seed for test data
        
        # Base price based on symbol
        base_prices = {'BTCUSDT': 45000, 'ETHUSDT': 2800, 'SOLUSDT': 120}
        base_price = base_prices.get(symbol, 1000)
        
        # Generate price movements
        returns = np.random.normal(0.0001, 0.025, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create DataFrame with more realistic features
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.008, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.008, len(dates)))),
            'close': prices,
            'volume': np.random.uniform(800000, 12000000, len(dates)),
            'rsi': np.random.uniform(15, 85, len(dates)),
            'macd': np.random.uniform(-150, 150, len(dates)),
            'fundamental_confidence': np.random.uniform(0.3, 0.95, len(dates)),
            'market_sentiment': np.random.uniform(0.2, 0.9, len(dates)),
            'regulatory_impact': np.random.uniform(0.1, 0.8, len(dates)),
            'tech_innovation': np.random.uniform(0.3, 0.95, len(dates)),
            'geopolitical_risk': np.random.uniform(0.05, 0.7, len(dates))
        }, index=dates)
        
        return data
    
    def load_trained_agent(self, model_path: str = None) -> CryptoRLAgent:
        """Load the trained agent."""
        logger.info("🤖 Loading trained agent...")
        
        if model_path is None:
            # Find latest model
            model_files = list(self.models_dir.glob("agent_*.pth"))
            if not model_files:
                logger.warning("⚠️ No trained model found, creating new agent...")
                agent = CryptoRLAgent(
                    settings=self.settings,
                    state_dim=50,  # Will be updated by environment
                    action_dim=3,  # Buy, Sell, Hold
                    learning_rate=3e-4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_param=0.2,
                    value_loss_coef=0.5,
                    entropy_coef=0.01,
                    max_grad_norm=0.5
                )
                return agent
            
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.settings.device)
        
        # Create agent with saved dimensions
        agent = CryptoRLAgent(
            settings=self.settings,
            state_dim=checkpoint.get('state_dim', 50),
            action_dim=checkpoint.get('action_dim', 3),
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5
        )
        
        agent.policy.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"✅ Agent loaded from {model_path}")
        return agent
    
    def run_single_symbol_backtests(self, agent: CryptoRLAgent, test_data: Dict[str, pd.DataFrame]) -> Dict[str, BacktestResult]:
        """Run individual backtests for each symbol."""
        logger.info("🔍 Running single-symbol backtests...")
        
        results = {}
        
        for symbol, data in test_data.items():
            if data.empty:
                continue
            
            logger.info(f"Testing {symbol}...")
            
            try:
                result = self.backtest_engine.run_single_backtest(
                    agent=agent,
                    test_data=data,
                    symbol=symbol,
                    initial_balance=self.initial_balance
                )
                
                results[symbol] = result
                logger.info(f"✅ {symbol}: {result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe")
                
            except Exception as e:
                logger.error(f"❌ Error testing {symbol}: {e}")
        
        return results
    
    def run_walk_forward_analysis(self, agent: CryptoRLAgent, test_data: Dict[str, pd.DataFrame]) -> Dict[str, List[BacktestResult]]:
        """Run walk-forward analysis for each symbol."""
        logger.info("🔄 Running walk-forward analysis...")
        
        walk_results = {}
        
        for symbol, data in test_data.items():
            if data.empty or len(data) < 252:  # Need at least 1 year of data
                continue
            
            logger.info(f"Walk-forward analysis for {symbol}...")
            
            try:
                results = self.backtest_engine.run_walk_forward_analysis(
                    agent=agent,
                    data=data,
                    symbol=symbol,
                    train_period=252,  # 1 year
                    test_period=63,    # 3 months
                    step_size=21       # 1 month
                )
                
                walk_results[symbol] = results
                logger.info(f"✅ {symbol}: {len(results)} walk-forward periods")
                
            except Exception as e:
                logger.error(f"❌ Error in walk-forward for {symbol}: {e}")
        
        return walk_results
    
    def run_portfolio_backtest(self, agent: CryptoRLAgent, test_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Run portfolio-level backtest with all symbols."""
        logger.info("💼 Running portfolio backtest...")
        
        # Combine all test data
        all_data = []
        for symbol, data in test_data.items():
            if not data.empty:
                data = data.copy()
                data['symbol'] = symbol
                all_data.append(data)
        
        if not all_data:
            logger.error("❌ No data available for portfolio backtest")
            return None
        
        combined_data = pd.concat(all_data)
        combined_data = combined_data.sort_index()
        
        # Run portfolio backtest
        try:
            result = self.backtest_engine.run_portfolio_backtest(
                agent=agent,
                data=combined_data,
                symbols=self.symbols,
                initial_balance=self.initial_balance
            )
            
            logger.info(f"✅ Portfolio backtest: {result.total_return:.2%} return")
            return result
            
        except Exception as e:
            logger.error(f"❌ Portfolio backtest failed: {e}")
            return None
    
    def calculate_risk_metrics(self, results: Dict[str, BacktestResult]) -> Dict:
        """Calculate comprehensive risk metrics."""
        logger.info("📊 Calculating risk metrics...")
        
        risk_metrics = {}
        
        for symbol, result in results.items():
            risk_metrics[symbol] = {
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'var_95': result.var_95,
                'var_99': result.var_99,
                'calmar_ratio': result.calmar_ratio,
                'sortino_ratio': result.sortino_ratio,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'profit_factor': result.profit_factor,
                'avg_trade_return': result.avg_trade_return
            }
        
        return risk_metrics
    
    def generate_performance_report(self, 
                                  single_results: Dict[str, BacktestResult],
                                  walk_results: Dict[str, List[BacktestResult]],
                                  portfolio_result: BacktestResult) -> str:
        """Generate comprehensive performance report."""
        logger.info("📋 Generating performance report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.results_dir / f"report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(single_results)
        
        # Create report
        report = f"""# CryptoRL Backtesting Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

### Individual Symbol Performance

| Symbol | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|--------|--------------|--------------|--------------|----------|--------|
"""
        
        for symbol, metrics in risk_metrics.items():
            report += f"| {symbol} | {metrics['total_return']:.2%} | {metrics['sharpe_ratio']:.2f} | {metrics['max_drawdown']:.2%} | {metrics['win_rate']:.2%} | {int(metrics['total_trades'])} |\n"
        
        if portfolio_result:
            report += f"""
### Portfolio Performance
- **Total Return**: {portfolio_result.total_return:.2%}
- **Sharpe Ratio**: {portfolio_result.sharpe_ratio:.2f}
- **Max Drawdown**: {portfolio_result.max_drawdown:.2%}
- **Win Rate**: {portfolio_result.win_rate:.2%}
- **Total Trades**: {portfolio_result.total_trades}

"""
        
        # Walk-forward analysis
        if walk_results:
            report += "## Walk-Forward Analysis\n\n"
            for symbol, walk_periods in walk_results.items():
                if walk_periods:
                    avg_return = np.mean([r.total_return for r in walk_periods])
                    avg_sharpe = np.mean([r.sharpe_ratio for r in walk_periods])
                    report += f"**{symbol}** - {len(walk_periods)} periods, Avg Return: {avg_return:.2%}, Avg Sharpe: {avg_sharpe:.2f}\n\n"
        
        report += "## Risk Analysis\n\n"
        report += "### Value at Risk\n"
        for symbol, metrics in risk_metrics.items():
            report += f"- **{symbol}**: VaR(95%) = {metrics['var_95']:.2%}, VaR(99%) = {metrics['var_99']:.2%}\n"
        
        report += "\n### Risk-Adjusted Returns\n"
        for symbol, metrics in risk_metrics.items():
            report += f"- **{symbol}**: Calmar = {metrics['calmar_ratio']:.2f}, Sortino = {metrics['sortino_ratio']:.2f}\n"
        
        # Save report
        report_path = report_dir / "backtest_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save detailed metrics
        metrics_df = pd.DataFrame.from_dict(risk_metrics, orient='index')
        metrics_df.to_csv(report_dir / "risk_metrics.csv")
        
        # Save results JSON
        results_data = {}
        for symbol, result in single_results.items():
            results_data[symbol] = {
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'final_value': result.final_value
            }
        
        with open(report_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"📋 Report saved to {report_dir}")
        return str(report_dir)
    
    def create_visualizations(self, results: Dict[str, BacktestResult], report_dir: str):
        """Create performance visualizations."""
        logger.info("📊 Creating visualizations...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CryptoRL Backtesting Results', fontsize=16)
        
        # Returns comparison
        symbols = list(results.keys())
        returns = [results[s].total_return for s in symbols]
        
        axes[0, 0].bar(symbols, returns)
        axes[0, 0].set_title('Total Returns by Symbol')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sharpe ratio comparison
        sharpe_ratios = [results[s].sharpe_ratio for s in symbols]
        axes[0, 1].bar(symbols, sharpe_ratios, color='orange')
        axes[0, 1].set_title('Sharpe Ratios by Symbol')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Max drawdown comparison
        max_drawdowns = [results[s].max_drawdown for s in symbols]
        axes[1, 0].bar(symbols, max_drawdowns, color='red')
        axes[1, 0].set_title('Max Drawdown by Symbol')
        axes[1, 0].set_ylabel('Max Drawdown')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Win rate comparison
        win_rates = [results[s].win_rate for s in symbols]
        axes[1, 1].bar(symbols, win_rates, color='green')
        axes[1, 1].set_title('Win Rate by Symbol')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{report_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Visualizations saved to {report_dir}")
    
    def run(self, model_path: str = None) -> bool:
        """Run comprehensive backtesting pipeline."""
        logger.info("🎯 Starting Comprehensive Backtesting Pipeline")
        
        try:
            # Step 1: Load test data
            test_data = self.load_test_data()
            
            # Step 2: Load trained agent
            agent = self.load_trained_agent(model_path)
            
            # Step 3: Run single-symbol backtests
            single_results = self.run_single_symbol_backtests(agent, test_data)
            
            if not single_results:
                logger.error("❌ No backtest results generated")
                return False
            
            # Step 4: Run walk-forward analysis
            walk_results = self.run_walk_forward_analysis(agent, test_data)
            
            # Step 5: Run portfolio backtest
            portfolio_result = self.run_portfolio_backtest(agent, test_data)
            
            # Step 6: Generate report
            report_dir = self.generate_performance_report(
                single_results, walk_results, portfolio_result
            )
            
            # Step 7: Create visualizations
            self.create_visualizations(single_results, report_dir)
            
            # Print summary
            print("\n" + "="*60)
            print("🎯 BACKTESTING SUMMARY")
            print("="*60)
            
            for symbol, result in single_results.items():
                print(f"{symbol}: {result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe")
            
            if portfolio_result:
                print(f"\nPortfolio: {portfolio_result.total_return:.2%} return, {portfolio_result.sharpe_ratio:.2f} Sharpe")
            
            print(f"\n📊 Report saved to: {report_dir}")
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backtesting failed: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive backtesting")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--symbols", nargs="+", default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize settings
    settings = Settings()
    
    # Create backtest pipeline
    pipeline = ComprehensiveBacktestPipeline(settings)
    pipeline.symbols = args.symbols
    
    # Run backtesting
    success = pipeline.run(args.model)
    
    if success:
        print("\n🎉 Backtesting completed successfully!")
        print("Check results in the results/backtests/ directory")
    else:
        print("\n❌ Backtesting failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()