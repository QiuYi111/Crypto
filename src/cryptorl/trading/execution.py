"""Comprehensive trading execution system for Binance."""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path
import time
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException

from ..config.settings import Settings
from ..risk_management.risk_manager import RiskManager, RiskMetrics


@dataclass
class Order:
    """Trading order."""
    symbol: str
    side: str  # BUY or SELL
    order_type: str  # MARKET, LIMIT, STOP_LOSS, etc.
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: str  # LONG or SHORT
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float = 1.0
    margin: float = 0.0


class BinanceTrader:
    """Binance trading execution system."""
    
    def __init__(self, settings: Settings, risk_manager: RiskManager):
        self.settings = settings
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize Binance client
        self.client = Client(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            testnet=settings.binance_testnet
        )
        
        # Trading parameters
        self.max_retry_attempts = 3
        self.retry_delay = 1.0
        self.order_timeout = 30.0
        
        # Position tracking
        self.open_positions = {}
        self.order_history = []
        
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place an order on Binance."""
        
        try:
            # Validate order with risk manager
            risk_metrics = self.risk_manager.evaluate_risk(
                symbol=order.symbol,
                current_price=await self.get_current_price(order.symbol),
                position_size=order.quantity
            )
            
            should_trade, reason = self.risk_manager.should_enter_position(
                symbol=order.symbol,
                proposed_size=order.quantity,
                current_price=await self.get_current_price(order.symbol),
                risk_metrics=risk_metrics
            )
            
            if not should_trade:
                self.logger.warning(f"Order rejected by risk manager: {reason}")
                return {"status": "rejected", "reason": reason}
            
            # Place order
            order_params = {
                'symbol': order.symbol,
                'side': order.side,
                'type': order.order_type,
                'quantity': order.quantity,
                'timeInForce': order.time_in_force
            }
            
            if order.price:
                order_params['price'] = order.price
            if order.stop_price:
                order_params['stopPrice'] = order.stop_price
            if order.client_order_id:
                order_params['newClientOrderId'] = order.client_order_id
            
            # Execute with retry
            result = await self._execute_with_retry(
                self.client.futures_create_order,
                **order_params
            )
            
            # Log order
            self._log_order(order, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"Error getting price: {e}")
            return 0.0
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get current account balance."""
        
        try:
            account = self.client.futures_account()
            balances = {}
            
            for asset in account.get('assets', []):
                balance = float(asset.get('walletBalance', 0))
                if balance > 0:
                    balances[asset.get('asset', '')] = balance
            
            return balances
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return {}
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        
        try:
            account = self.client.futures_account()
            positions = account.get('positions', [])
            
            for pos in positions:
                if pos['symbol'] == symbol:
                    quantity = float(pos.get('positionAmt', 0))
                    if abs(quantity) > 0:
                        return Position(
                            symbol=symbol,
                            side="LONG" if quantity > 0 else "SHORT",
                            quantity=abs(quantity),
                            entry_price=float(pos.get('entryPrice', 0)),
                            current_price=float(pos.get('markPrice', 0)),
                            unrealized_pnl=float(pos.get('unRealizedProfit', 0)),
                            leverage=float(pos.get('leverage', 1.0))
                        )
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting position: {e}")
            return None
    
    def _log_order(self, order: Order, result: Dict[str, Any]):
        """Log order execution."""
        
        order_log = {
            'timestamp': datetime.now(),
            'symbol': order.symbol,
            'side': order.side,
            'type': order.order_type,
            'quantity': order.quantity,
            'price': order.price,
            'status': result.get('status', 'unknown'),
            'order_id': result.get('orderId', 'N/A')
        }
        
        self.order_history.append(order_log)
        
        # Save to file
        log_path = Path("./trading_logs")
        log_path.mkdir(exist_ok=True)
        
        with open(log_path / "order_history.json", "a") as f:
            json.dump(order_log, f, default=str)
            f.write("\n")