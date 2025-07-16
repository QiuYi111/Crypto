"""Risk management system for CryptoRL trading."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import pandas as pd
import numpy as np

from ..config.settings import Settings


@dataclass
class RiskMetrics:
    """Risk metrics for a position."""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    max_drawdown: float
    value_at_risk: float
    risk_score: float


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.position_limits = {
            "max_position_size": settings.rl.max_position_size,
            "max_leverage": settings.max_leverage,
            "max_total_exposure": 0.8,  # Maximum 80% of portfolio
            "max_daily_loss": 0.05,     # 5% daily loss limit
            "max_drawdown": 0.15        # 15% maximum drawdown
        }
        self.risk_history = []
        
    def calculate_position_risk(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        current_price: float,
        portfolio_value: float
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a position."""
        
        # Calculate P&L
        unrealized_pnl = (current_price - entry_price) * position_size
        unrealized_pnl_percent = (unrealized_pnl / (entry_price * position_size)) * 100
        
        # Calculate drawdown (simplified)
        max_drawdown = abs(unrealized_pnl_percent) if unrealized_pnl < 0 else 0
        
        # Value at Risk (simplified 95% VaR using normal distribution)
        # In production, use historical VaR or GARCH models
        volatility = 0.02  # 2% daily volatility assumption
        var_95 = position_size * current_price * volatility * 1.645
        
        # Risk score (0-100, higher = more risky)
        risk_score = self._calculate_risk_score(
            position_size=position_size,
            unrealized_pnl=unrealized_pnl,
            portfolio_value=portfolio_value,
            max_drawdown=max_drawdown
        )
        
        return RiskMetrics(
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=unrealized_pnl_percent,
            max_drawdown=max_drawdown,
            value_at_risk=var_95,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(
        self,
        position_size: float,
        unrealized_pnl: float,
        portfolio_value: float,
        max_drawdown: float
    ) -> float:
        """Calculate overall risk score."""
        
        # Position size risk (0-40 points)
        position_risk = min((position_size / portfolio_value) * 100, 40)
        
        # P&L risk (0-30 points)
        pnl_risk = min(abs(unrealized_pnl) / portfolio_value * 100, 30)
        
        # Drawdown risk (0-30 points)
        drawdown_risk = min(max_drawdown * 2, 30)
        
        return position_risk + pnl_risk + drawdown_risk
    
    def check_position_limits(
        self,
        symbol: str,
        proposed_position: float,
        current_portfolio: Dict[str, Any],
        current_price: float
    ) -> Tuple[bool, str]:
        """Check if proposed position is within risk limits."""
        
        portfolio_value = current_portfolio.get("total_value", 0)
        current_exposure = current_portfolio.get("total_exposure", 0)
        
        # Check maximum position size
        max_position_value = portfolio_value * self.position_limits["max_position_size"]
        max_position = max_position_value / current_price
        
        if abs(proposed_position) > max_position:
            return False, f"Position size exceeds limit: {max_position}"
        
        # Check total exposure
        new_exposure = current_exposure + abs(proposed_position * current_price)
        max_exposure = portfolio_value * self.position_limits["max_total_exposure"]
        
        if new_exposure > max_exposure:
            return False, f"Total exposure exceeds limit: {max_exposure}"
        
        # Check leverage
        leverage = new_exposure / portfolio_value
        if leverage > self.position_limits["max_leverage"]:
            return False, f"Leverage exceeds limit: {self.position_limits['max_leverage']}"
        
        return True, "Position within limits"
    
    def calculate_position_size(
        self,
        signal_confidence: float,
        portfolio_value: float,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.05
    ) -> float:
        """Calculate optimal position size using Kelly Criterion."""
        
        # Basic Kelly Criterion implementation
        # Simplified for demonstration
        
        # Risk per trade (2% of portfolio)
        dollar_risk = portfolio_value * risk_per_trade
        
        # Position size based on stop loss
        position_value = dollar_risk / stop_loss_pct
        
        # Adjust by confidence
        confidence_factor = min(signal_confidence, 0.8)  # Cap at 80%
        adjusted_position = position_value * confidence_factor
        
        return adjusted_position
    
    def should_stop_loss(
        self,
        position: Dict[str, Any],
        current_price: float,
        entry_price: float
    ) -> bool:
        """Determine if stop loss should be triggered."""
        
        if not position or not position.get("size", 0):
            return False
        
        # Calculate loss percentage
        loss_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Check stop loss trigger
        if position.get("side") == "BUY":
            # Long position
            if loss_pct <= -self.position_limits["max_daily_loss"] * 100:
                return True
        else:
            # Short position
            if loss_pct >= self.position_limits["max_daily_loss"] * 100:
                return True
        
        return False
    
    def should_take_profit(
        self,
        position: Dict[str, Any],
        current_price: float,
        entry_price: float,
        profit_target: float = 0.03
    ) -> bool:
        """Determine if take profit should be triggered."""
        
        if not position or not position.get("size", 0):
            return False
        
        # Calculate profit percentage
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Check take profit trigger
        if position.get("side") == "BUY":
            # Long position
            if profit_pct >= profit_target * 100:
                return True
        else:
            # Short position
            if profit_pct <= -profit_target * 100:
                return True
        
        return False
    
    def get_portfolio_risk_summary(
        self,
        positions: List[Dict[str, Any]],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary."""
        
        if not positions:
            return {
                "total_exposure": 0,
                "total_risk": 0,
                "max_drawdown": 0,
                "risk_score": 0,
                "is_healthy": True
            }
        
        total_exposure = sum(abs(pos["size"] * pos["current_price"]) for pos in positions)
        total_risk = sum(pos.get("var_95", 0) for pos in positions)
        max_drawdown = max(pos.get("max_drawdown", 0) for pos in positions)
        avg_risk_score = np.mean([pos.get("risk_score", 0) for pos in positions])
        
        # Portfolio health check
        exposure_ratio = total_exposure / portfolio_value
        is_healthy = (
            exposure_ratio <= self.position_limits["max_total_exposure"] and
            max_drawdown <= self.position_limits["max_drawdown"]
        )
        
        return {
            "total_exposure": total_exposure,
            "total_risk": total_risk,
            "max_drawdown": max_drawdown,
            "risk_score": avg_risk_score,
            "exposure_ratio": exposure_ratio,
            "is_healthy": is_healthy,
            "positions_count": len(positions)
        }
    
    def generate_risk_report(self, positions: List[Dict[str, Any]]) -> str:
        """Generate comprehensive risk report."""
        
        if not positions:
            return "No active positions."
        
        portfolio_value = sum(pos["size"] * pos["current_price"] for pos in positions)
        risk_summary = self.get_portfolio_risk_summary(positions, portfolio_value)
        
        report = f"""# Risk Management Report

## Portfolio Summary
- Total Exposure: ${risk_summary['total_exposure']:.2f}
- Portfolio Value: ${portfolio_value:.2f}
- Exposure Ratio: {risk_summary['exposure_ratio']:.2%}
- Max Drawdown: {risk_summary['max_drawdown']:.2%}
- Risk Score: {risk_summary['risk_score']:.1f}/100
- Portfolio Health: {'✅ HEALTHY' if risk_summary['is_healthy'] else '❌ UNHEALTHY'}

## Position Details
"""
        
        for pos in positions:
            risk_metrics = self.calculate_position_risk(
                symbol=pos["symbol"],
                position_size=pos["size"],
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                portfolio_value=portfolio_value
            )
            
            report += f"""
### {pos['symbol']}
- Position Size: {pos['size']:.6f}
- Entry Price: ${pos['entry_price']:.2f}
- Current Price: ${pos['current_price']:.2f}
- Unrealized P&L: ${risk_metrics.unrealized_pnl:.2f} ({risk_metrics.unrealized_pnl_percent:.2f}%)
- Value at Risk: ${risk_metrics.value_at_risk:.2f}
- Risk Score: {risk_metrics.risk_score:.1f}/100
"""
        
        return report
    
    def log_risk_event(self, event_type: str, details: Dict[str, Any]):
        """Log risk-related events."""
        
        risk_event = {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "details": details
        }
        
        self.risk_history.append(risk_event)
        
        logger.info(f"Risk event: {event_type} - {details}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check risk management system health."""
        
        return {
            "status": "healthy",
            "risk_limits_configured": bool(self.position_limits),
            "positions_monitored": len(self.risk_history),
            "last_update": datetime.utcnow().isoformat()
        }