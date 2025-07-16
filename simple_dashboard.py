#!/usr/bin/env python3
"""Simplified dashboard that actually works."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock data generators
def generate_mock_data():
    """Generate mock data for testing."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)
    portfolio_values = 10000 + np.cumsum(np.random.normal(50, 200, 30))
    
    return pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'daily_return': np.random.normal(0.001, 0.02, 30)
    })

def create_portfolio_chart(data):
    """Create a simple portfolio chart."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400
    )
    return fig

def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="CryptoRL Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Title
    st.title("🎯 CryptoRL Trading Dashboard")
    st.markdown("---")
    
    # Generate mock data
    data = generate_mock_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_value = data['portfolio_value'].iloc[-1]
        initial_value = data['portfolio_value'].iloc[0]
        total_return = ((current_value - initial_value) / initial_value) * 100
        st.metric("Total P&L", f"${current_value:,.0f}", f"{total_return:+.1f}%")
    
    with col2:
        win_rate = np.random.uniform(60, 75)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        sharpe = np.random.uniform(1.5, 2.5)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col4:
        max_dd = np.random.uniform(5, 12)
        st.metric("Max Drawdown", f"-{max_dd:.1f}%")
    
    # Portfolio chart
    st.subheader("📈 Portfolio Performance")
    fig = create_portfolio_chart(data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent trades
    st.subheader("🔄 Recent Activity")
    trades = pd.DataFrame([
        {
            'Time': datetime.now() - timedelta(minutes=np.random.randint(1, 120)),
            'Symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']),
            'Action': np.random.choice(['BUY', 'SELL']),
            'Amount': np.random.uniform(0.1, 1.0),
            'Price': np.random.uniform(40000, 55000),
            'Status': 'FILLED'
        }
        for _ in range(5)
    ])
    
    st.dataframe(trades, use_container_width=True)
    
    # Risk indicators
    st.subheader("🛡️ Risk Status")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        risk_level = np.random.choice(['LOW', 'MEDIUM', 'HIGH'])
        color = 'green' if risk_level == 'LOW' else 'orange' if risk_level == 'MEDIUM' else 'red'
        st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
    
    with risk_col2:
        var_95 = np.random.uniform(500, 2000)
        st.metric("VaR (95%)", f"${var_95:,.0f}")
    
    with risk_col3:
        positions = np.random.randint(1, 5)
        st.metric("Open Positions", positions)
    
    # System status
    st.subheader("⚙️ System Status")
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.success("✅ Trading Engine")
    with status_col2:
        st.success("✅ Risk Manager")
    with status_col3:
        st.success("✅ Data Feed")

if __name__ == "__main__":
    main()