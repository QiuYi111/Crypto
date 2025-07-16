"""Real-time monitoring dashboard for CryptoRL trading system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import time
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging

from cryptorl.backtesting.engine import BacktestingEngine, BacktestResult
from cryptorl.risk_management.risk_manager import RiskManager, RiskLevel
from cryptorl.trading.execution import BinanceTrader
from cryptorl.config.settings import Settings


class MonitoringDashboard:
    """Real-time monitoring dashboard for crypto trading system."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.risk_manager = RiskManager(settings)
        
        # Data storage
        self.portfolio_history = []
        self.trade_history = []
        self.risk_events = []
        self.performance_metrics = {}
        
        # Dashboard configuration
        self.update_interval = 5  # Default 5 second update interval
        
    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        
        try:
            st.set_page_config(
                page_title="CryptoRL Monitoring Dashboard",
                page_icon="📊",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        except Exception as e:
            # Fallback for testing
            st.set_page_config(
                page_title="CryptoRL Dashboard",
                layout="wide"
            )
        
        # Custom CSS
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
        }
        .risk-low { color: green; }
        .risk-medium { color: orange; }
        .risk-high { color: red; }
        .risk-critical { color: darkred; }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("🎯 CryptoRL Dashboard")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Portfolio", "Trading", "Risk Management", "Performance", "Logs"]
        )
        
        # Settings
        st.sidebar.header("Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 60, 10)
        
        # Main content
        if page == "Overview":
            self._render_overview_page()
        elif page == "Portfolio":
            self._render_portfolio_page()
        elif page == "Trading":
            self._render_trading_page()
        elif page == "Risk Management":
            self._render_risk_page()
        elif page == "Performance":
            self._render_performance_page()
        elif page == "Logs":
            self._render_logs_page()
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def _render_overview_page(self):
        """Render overview page."""
        
        st.title("📊 CryptoRL System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total P&L",
                "$12,450.00",
                "$1,234.00",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Win Rate",
                "68.5%",
                "+2.3%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                "2.15",
                "+0.15",
                delta_color="normal"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                "-8.2%",
                "-0.5%",
                delta_color="inverse"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Value Over Time")
            fig = self._create_portfolio_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Daily Returns Distribution")
            fig = self._create_returns_distribution()
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent trades
        st.subheader("Recent Trades")
        recent_trades = self._get_recent_trades()
        if not recent_trades.empty:
            st.dataframe(recent_trades.tail(10), use_container_width=True)
        else:
            st.info("No trades yet")
    
    def _render_portfolio_page(self):
        """Render portfolio page."""
        
        st.title("💼 Portfolio Management")
        
        # Current positions
        st.subheader("Current Positions")
        positions_df = self._get_current_positions()
        
        if not positions_df.empty:
            # Position summary
            total_value = positions_df['current_value'].sum()
            total_pnl = positions_df['unrealized_pnl'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Position Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total Unrealized P&L", f"${total_pnl:,.2f}")
            with col3:
                st.metric("Number of Positions", len(positions_df))
            
            # Positions table
            st.dataframe(positions_df, use_container_width=True)
            
            # Position allocation pie chart
            fig = px.pie(
                positions_df,
                values='current_value',
                names='symbol',
                title='Portfolio Allocation'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active positions")
    
    def _render_trading_page(self):
        """Render trading page."""
        
        st.title("📈 Trading Activity")
        
        # Trading controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Data")
            symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            
            # Mock current price
            current_price = np.random.uniform(40000, 60000)
            st.metric("Current Price", f"${current_price:,.2f}")
            
            # Price chart
            price_data = self._generate_mock_price_data(symbol)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_data['timestamp'],
                y=price_data['price'],
                mode='lines',
                name='Price'
            ))
            fig.update_layout(title=f"{symbol} Price Chart")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Trading Signals")
            
            # Mock signals
            signals = self._get_mock_signals()
            if signals:
                for signal in signals:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.write(signal['symbol'])
                    with col_b:
                        st.write(f"{signal['action']} {signal['quantity']}")
                    with col_c:
                        st.write(f"Confidence: {signal['confidence']:.2f}")
            
            # Manual trading
            st.subheader("Manual Trading")
            manual_symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], key="manual")
            manual_side = st.selectbox("Side", ["BUY", "SELL"])
            manual_quantity = st.number_input("Quantity", min_value=0.001, max_value=1.0, value=0.1)
            
            if st.button("Place Order"):
                st.success(f"Order placed: {manual_side} {manual_quantity} {manual_symbol}")
    
    def _render_risk_page(self):
        """Render risk management page."""
        
        st.title("🛡️ Risk Management")
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_level = "LOW"
            st.metric("Risk Level", risk_level, delta_color="off")
        
        with col2:
            st.metric("Portfolio VaR (95%)", "-$1,234.56")
        
        with col3:
            st.metric("Current Drawdown", "-3.2%")
        
        with col4:
            st.metric("Max Leverage", "2.5x")
        
        # Risk charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Heatmap")
            risk_data = self._generate_risk_heatmap()
            fig = px.imshow(
                risk_data,
                labels=dict(x="Asset", y="Risk Factor", color="Risk Level"),
                title="Risk Assessment Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Drawdown Analysis")
            dd_data = self._generate_drawdown_data()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dd_data['date'],
                y=dd_data['drawdown'],
                fill='tozeroy',
                name='Drawdown'
            ))
            fig.update_layout(title="Portfolio Drawdown")
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk events
        st.subheader("Risk Events")
        risk_events = self._get_risk_events()
        if not risk_events.empty:
            st.dataframe(risk_events, use_container_width=True)
        else:
            st.info("No recent risk events")
    
    def _render_performance_page(self):
        """Render performance page."""
        
        st.title("📊 Performance Analytics")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", "+24.5%")
            st.metric("Annualized Return", "+18.2%")
        
        with col2:
            st.metric("Sharpe Ratio", "2.15")
            st.metric("Sortino Ratio", "1.89")
        
        with col3:
            st.metric("Max Drawdown", "-8.2%")
            st.metric("Calmar Ratio", "2.22")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cumulative Returns")
            returns_data = self._generate_returns_data()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=returns_data['date'],
                y=returns_data['cumulative_returns'],
                mode='lines',
                name='Cumulative Returns'
            ))
            fig.update_layout(title="Cumulative Returns Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Rolling Sharpe Ratio")
            sharpe_data = self._generate_sharpe_data()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sharpe_data['date'],
                y=sharpe_data['sharpe_ratio'],
                mode='lines',
                name='Rolling Sharpe'
            ))
            fig.update_layout(title="30-Day Rolling Sharpe Ratio")
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade statistics
        st.subheader("Trade Statistics")
        trade_stats = self._get_trade_statistics()
        st.dataframe(trade_stats, use_container_width=True)
    
    def _render_logs_page(self):
        """Render logs page."""
        
        st.title("📝 System Logs")
        
        # Log filters
        col1, col2 = st.columns(2)
        
        with col1:
            log_level = st.selectbox("Log Level", ["ALL", "INFO", "WARNING", "ERROR"])
        
        with col2:
            date_filter = st.date_input("Filter by Date")
        
        # Display logs
        logs = self._get_system_logs(log_level, date_filter)
        
        if logs:
            for log in logs:
                st.text(log)
        else:
            st.info("No logs found")
        
        # Export logs
        if st.button("Export Logs"):
            self._export_logs()
            st.success("Logs exported successfully")
    
    def _create_portfolio_chart(self) -> go.Figure:
        """Create portfolio value chart."""
        
        # Mock data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)
        values = 10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 30)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode='x'
        )
        
        return fig
    
    def _create_returns_distribution(self) -> go.Figure:
        """Create returns distribution chart."""
        
        # Mock data
        returns = np.random.normal(0.001, 0.02, 100)
        
        fig = px.histogram(
            x=returns,
            nbins=30,
            title="Daily Returns Distribution"
        )
        fig.update_layout(
            xaxis_title="Daily Return",
            yaxis_title="Frequency"
        )
        
        return fig
    
    def _get_current_positions(self) -> pd.DataFrame:
        """Get current positions."""
        
        # Mock data
        positions = [
            {
                'symbol': 'BTCUSDT',
                'side': 'LONG',
                'quantity': 0.5,
                'entry_price': 45000,
                'current_price': 52000,
                'current_value': 26000,
                'unrealized_pnl': 3500,
                'pnl_percentage': 15.56
            },
            {
                'symbol': 'ETHUSDT',
                'side': 'SHORT',
                'quantity': 2.0,
                'entry_price': 3200,
                'current_price': 3100,
                'current_value': 6200,
                'unrealized_pnl': 200,
                'pnl_percentage': 3.23
            }
        ]
        
        df = pd.DataFrame(positions)
        # Ensure numeric types
        df['quantity'] = pd.to_numeric(df['quantity'])
        df['entry_price'] = pd.to_numeric(df['entry_price'])
        df['current_price'] = pd.to_numeric(df['current_price'])
        df['current_value'] = pd.to_numeric(df['current_value'])
        df['unrealized_pnl'] = pd.to_numeric(df['unrealized_pnl'])
        df['pnl_percentage'] = pd.to_numeric(df['pnl_percentage'])
        return df
    
    def _get_recent_trades(self) -> pd.DataFrame:
        """Get recent trades."""
        
        # Mock data
        trades = [
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'symbol': 'BTCUSDT',
                'action': 'BUY',
                'quantity': 0.1,
                'price': 51000,
                'pnl': 500,
                'status': 'FILLED'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=5),
                'symbol': 'ETHUSDT',
                'action': 'SELL',
                'quantity': 0.5,
                'price': 3150,
                'pnl': 75,
                'status': 'FILLED'
            }
        ]
        
        df = pd.DataFrame(trades)
        # Ensure numeric types and datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['quantity'] = pd.to_numeric(df['quantity'])
        df['price'] = pd.to_numeric(df['price'])
        df['pnl'] = pd.to_numeric(df['pnl'])
        return df
    
    def _generate_mock_price_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock price data."""
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=100, freq='H')
        prices = 50000 + np.cumsum(np.random.normal(0, 100, 100))
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices
        })
    
    def _get_mock_signals(self) -> List[Dict[str, Any]]:
        """Get mock trading signals."""
        
        return [
            {
                'symbol': 'BTCUSDT',
                'action': 'BUY',
                'quantity': 0.1,
                'confidence': 0.85
            },
            {
                'symbol': 'ETHUSDT',
                'action': 'SELL',
                'quantity': 0.5,
                'confidence': 0.72
            }
        ]
    
    def _generate_risk_heatmap(self) -> np.ndarray:
        """Generate risk heatmap data."""
        
        np.random.seed(42)
        return np.random.rand(5, 3)
    
    def _generate_drawdown_data(self) -> pd.DataFrame:
        """Generate drawdown data."""
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)
        drawdown = np.cumsum(np.random.normal(-0.001, 0.01, 30))
        
        return pd.DataFrame({
            'date': dates,
            'drawdown': drawdown
        })
    
    def _generate_returns_data(self) -> pd.DataFrame:
        """Generate returns data."""
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)
        returns = np.cumprod(1 + np.random.normal(0.001, 0.02, 30))
        
        return pd.DataFrame({
            'date': dates,
            'cumulative_returns': (returns - 1) * 100
        })
    
    def _generate_sharpe_data(self) -> pd.DataFrame:
        """Generate Sharpe ratio data."""
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)
        sharpe = 2.0 + np.cumsum(np.random.normal(0, 0.1, 30))
        
        return pd.DataFrame({
            'date': dates,
            'sharpe_ratio': sharpe
        })
    
    def _get_trade_statistics(self) -> pd.DataFrame:
        """Get trade statistics."""
        
        stats = [
            {
                'Metric': 'Total Trades',
                'Value': 156
            },
            {
                'Metric': 'Win Rate',
                'Value': 68.5
            },
            {
                'Metric': 'Average Win',
                'Value': 245.50
            },
            {
                'Metric': 'Average Loss',
                'Value': -89.20
            },
            {
                'Metric': 'Profit Factor',
                'Value': 2.75
            },
            {
                'Metric': 'Average Hold Time',
                'Value': 2.3
            }
        ]
        
        return pd.DataFrame(stats)
    
    def _get_risk_events(self) -> pd.DataFrame:
        """Get risk events."""
        
        events = [
            {
                'timestamp': datetime.now() - timedelta(hours=1),
                'type': 'WARNING',
                'message': 'High volatility detected for BTC',
                'severity': 'MEDIUM'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=5),
                'type': 'INFO',
                'message': 'Risk limit updated for ETH',
                'severity': 'LOW'
            }
        ]
        
        df = pd.DataFrame(events)
        # Ensure timestamp column is datetime type for compatibility
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def _get_system_logs(self, level: str, date: datetime) -> List[str]:
        """Get system logs."""
        
        logs = [
            f"{datetime.now()} - INFO - System started successfully",
            f"{datetime.now()} - INFO - RL model loaded",
            f"{datetime.now()} - WARNING - High memory usage detected",
            f"{datetime.now()} - INFO - Trading session started"
        ]
        
        return logs
    
    def _export_logs(self):
        """Export logs to file."""
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'dashboard_status': 'healthy',
            'metrics': self.performance_metrics
        }
        
        log_path = Path("./dashboard_logs")
        log_path.mkdir(exist_ok=True)
        
        with open(log_path / "dashboard_export.json", "w") as f:
            json.dump(log_data, f, indent=2, default=str)


def launch_dashboard(settings: Settings):
    """Launch the monitoring dashboard."""
    
    dashboard = MonitoringDashboard(settings)
    dashboard.run_dashboard()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, './src')
    
    from cryptorl.config.settings import Settings
    
    settings = Settings()
    launch_dashboard(settings)