"""Comprehensive risk management system for crypto trading."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Current risk metrics for a position or portfolio."""
    
    # Position sizing
    position_size: float
    max_position_size: float
    leverage: float
    max_leverage: float
    
    # P&L metrics
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    pnl_percentage: float
    
    # Risk metrics
    var_95: float
    var_99: float
    max_drawdown: float
    current_drawdown: float
    
    # Volatility measures
    realized_volatility: float
    implied_volatility: float
    
    # Correlation measures
    correlation_with_market: float
    beta: float
    
    # Liquidity measures
    bid_ask_spread: float
    volume_ratio: float
    
    # Overall risk level
    risk_level: RiskLevel
    risk_score: float


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.max_position_size = settings.rl_max_position_size
        self.max_portfolio_drawdown = 0.2  # 20% max drawdown
        self.max_daily_loss = 1000.0  # $1000 max daily loss
        self.max_leverage = 10.0  # Default max leverage
        self.var_limit = 0.05  # 5% VaR limit
        
        # Tracking
        self.position_history = []
        self.risk_events = []
        self.current_positions = {}
        
    def evaluate_risk(self, 
                     symbol: str, 
                     current_price: float,
                     position_size: float = 0.0,
                     entry_price: float = 0.0,
                     market_data: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """Evaluate risk metrics for a position."""
        
        # Calculate basic metrics
        unrealized_pnl = (current_price - entry_price) * position_size if position_size != 0 else 0.0
        pnl_percentage = (unrealized_pnl / (entry_price * abs(position_size))) * 100 if position_size != 0 else 0.0
        
        # Calculate risk metrics
        var_95 = self._calculate_var(symbol, market_data, 0.05) if market_data is not None else 0.0
        var_99 = self._calculate_var(symbol, market_data, 0.01) if market_data is not None else 0.0
        
        # Volatility measures
        realized_vol = self._calculate_realized_volatility(symbol, market_data) if market_data is not None else 0.0
        
        # Drawdown calculations
        max_dd, current_dd = self._calculate_drawdowns(symbol)
        
        # Risk scoring
        risk_score = self._calculate_risk_score(
            position_size, unrealized_pnl, var_95, max_dd, realized_vol
        )
        risk_level = self._determine_risk_level(risk_score)
        
        return RiskMetrics(
            position_size=position_size,
            max_position_size=self.max_position_size,
            leverage=abs(position_size) / self.settings.risk_initial_balance,
            max_leverage=self.max_leverage,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=0.0,
            total_pnl=unrealized_pnl,
            pnl_percentage=pnl_percentage,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            realized_volatility=realized_vol,
            implied_volatility=realized_vol,
            correlation_with_market=0.0,
            beta=1.0,
            bid_ask_spread=0.001,
            volume_ratio=1.0,
            risk_level=risk_level,
            risk_score=risk_score
        )
    
    def should_enter_position(self, 
                            symbol: str, 
                            proposed_size: float,
                            current_price: float,
                            risk_metrics: RiskMetrics) -> Tuple[bool, str]:
        """Determine if a new position should be entered."""
        
        # Check position size limits
        if abs(proposed_size) > self.max_position_size:
            return False, f"Position size {proposed_size} exceeds limit {self.max_position_size}"
        
        # Check leverage limits
        leverage = abs(proposed_size) / self.settings.risk_initial_balance
        if leverage > self.max_leverage:
            return False, f"Leverage {leverage:.2f}x exceeds limit {self.max_leverage}x"
        
        # Check portfolio drawdown
        if risk_metrics.current_drawdown > self.max_portfolio_drawdown:
            return False, f"Current drawdown {risk_metrics.current_drawdown:.2%} exceeds limit {self.max_portfolio_drawdown:.2%}"
        
        # Check VaR limit
        if abs(risk_metrics.var_95) > self.var_limit:
            return False, f"VaR(95%) {risk_metrics.var_95:.2%} exceeds limit {self.var_limit:.2%}"
        
        # Check risk level
        if risk_metrics.risk_level == RiskLevel.CRITICAL:
            return False, "Risk level is critical"
        
        return True, "Risk check passed"
    
    def should_exit_position(self, 
                           symbol: str,
                           risk_metrics: RiskMetrics,
                           stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None) -> Tuple[bool, str]:
        """Determine if a position should be exited."""
        
        # Stop loss check
        if stop_loss and risk_metrics.pnl_percentage <= -abs(stop_loss):
            return True, f"Stop loss triggered: {risk_metrics.pnl_percentage:.2f}% <= {stop_loss}%"
        
        # Take profit check
        if take_profit and risk_metrics.pnl_percentage >= abs(take_profit):
            return True, f"Take profit triggered: {risk_metrics.pnl_percentage:.2f}% >= {take_profit}%"
        
        # Risk level check
        if risk_metrics.risk_level == RiskLevel.CRITICAL:
            return True, "Risk level critical - emergency exit"
        
        # Drawdown check
        if risk_metrics.current_drawdown > self.max_portfolio_drawdown:
            return True, f"Drawdown {risk_metrics.current_drawdown:.2%} exceeds limit {self.max_portfolio_drawdown:.2%}"
        
        return False, "Hold position"
    
    def calculate_position_size(self, 
                              symbol: str,
                              confidence: float,
                              volatility: float,
                              account_balance: float) -> float:
        """Calculate optimal position size using Kelly Criterion with risk adjustments."""
        
        # Base Kelly Criterion
        edge = confidence * 0.1  # Assume 10% edge at full confidence
        kelly_fraction = edge / (volatility ** 2) if volatility > 0 else 0.0
        
        # Risk-adjusted Kelly (use 25% of Kelly to be conservative)
        adjusted_kelly = kelly_fraction * 0.25
        
        # Position size
        position_size = adjusted_kelly * account_balance
        
        # Apply risk limits
        position_size = min(position_size, self.max_position_size)
        
        return position_size
    
    def _calculate_var(self, symbol: str, market_data: pd.DataFrame, confidence: float) -> float:
        """Calculate Value at Risk for a position."""
        
        if market_data is None or len(market_data) < 30:
            return 0.0
        
        # Calculate daily returns
        returns = market_data['close'].pct_change().dropna()
        
        # Calculate VaR
        var = np.percentile(returns, confidence * 100)
        
        return abs(var)
    
    def _calculate_realized_volatility(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Calculate realized volatility."""
        
        if market_data is None or len(market_data) < 30:
            return 0.0
        
        returns = market_data['close'].pct_change().dropna()
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_drawdowns(self, symbol: str) -> Tuple[float, float]:
        """Calculate maximum and current drawdown."""
        
        # Simplified implementation - would use actual portfolio history
        max_drawdown = 0.0
        current_drawdown = 0.0
        
        return max_drawdown, current_drawdown
    
    def _calculate_risk_score(self, 
                            position_size: float,
                            unrealized_pnl: float,
                            var_95: float,
                            max_drawdown: float,
                            volatility: float) -> float:
        """Calculate overall risk score (0-100, higher = more risky)."""
        
        score = 0.0
        
        # Position size risk (0-25 points)
        position_ratio = abs(position_size) / self.max_position_size
        score += min(position_ratio * 25, 25)
        
        # P&L risk (0-20 points)
        pnl_risk = abs(unrealized_pnl) / self.settings.risk_initial_balance * 100
        score += min(pnl_risk * 2, 20)
        
        # VaR risk (0-20 points)
        var_risk = abs(var_95) * 100
        score += min(var_risk * 4, 20)
        
        # Drawdown risk (0-20 points)
        dd_risk = abs(max_drawdown) * 100
        score += min(dd_risk * 2, 20)
        
        # Volatility risk (0-15 points)
        vol_risk = volatility * 100
        score += min(vol_risk * 0.3, 15)
        
        return min(score, 100)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score."""
        
        if risk_score < 25:
            return RiskLevel.LOW
        elif risk_score < 50:
            return RiskLevel.MEDIUM
        elif risk_score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk status."""
        
        return {
            'total_positions': len(self.current_positions),
            'max_position_size': self.max_position_size,
            'max_portfolio_drawdown': self.max_portfolio_drawdown,
            'max_daily_loss': self.max_daily_loss,
            'max_leverage': self.max_leverage,
            'var_limit': self.var_limit,
            'risk_events': len(self.risk_events)
        }
    
    def log_risk_event(self, event_type: str, symbol: str, details: Dict[str, Any]):
        """Log a risk event."""
        
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'symbol': symbol,
            'details': details
        }
        
        self.risk_events.append(event)
        self.logger.warning(f"Risk event logged: {event_type} for {symbol}")
    
    def export_risk_report(self, output_path: str = "./risk_report.json"):
        """Export comprehensive risk report."""
        
        report = {
            'summary': self.get_risk_summary(),
            'risk_events': self.risk_events,
            'current_positions': self.current_positions,
            'limits': {
                'max_position_size': self.max_position_size,
                'max_portfolio_drawdown': self.max_portfolio_drawdown,
                'max_daily_loss': self.max_daily_loss,
                'max_leverage': self.max_leverage,
                'var_limit': self.var_limit
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path


class RiskMonitor:
    """Real-time risk monitoring system."""
    
    def __init__(self, risk_manager: RiskManager, alert_threshold: float = 0.8):
        self.risk_manager = risk_manager
        self.alert_threshold = alert_threshold
        self.active_alerts = []
        
    def monitor_position(self, symbol: str, current_price: float, position_size: float):
        """Monitor a position for risk alerts."""
        
        risk_metrics = self.risk_manager.evaluate_risk(symbol, current_price, position_size)
        
        # Check for alerts
        if risk_metrics.risk_score >= self.alert_threshold * 100:
            alert = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'risk_score': risk_metrics.risk_score,
                'risk_level': risk_metrics.risk_level.value,
                'message': f"High risk alert: score {risk_metrics.risk_score:.1f}/100"
            }
            
            self.active_alerts.append(alert)
            self.risk_manager.logger.warning(f"Risk alert: {alert['message']}")
            
        return risk_metrics
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get current active risk alerts."""
        
        # Remove expired alerts (older than 1 hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert['timestamp'] > cutoff
        ]
        
        return self.active_alerts