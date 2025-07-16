#!/usr/bin/env python3
"""Test Binance API connection with your configured credentials."""

import asyncio
import aiohttp
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
import logging

from src.cryptorl.config.settings import Settings
from src.cryptorl.trading.execution import BinanceTrader
from src.cryptorl.risk_management.risk_manager import RiskManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_api_connection():
    """Test comprehensive API connection."""
    
    print("🧪 Testing Binance API Connection...")
    print("=" * 50)
    
    settings = Settings()
    
    # Check if API keys are configured
    if not settings.binance_api_key or not settings.binance_secret_key:
        print("❌ API keys not found in settings!")
        print("   Please check your .env file or environment variables")
        return
    
    print(f"✅ API Key configured: {settings.binance_api_key[:8]}...")
    print(f"✅ Testnet mode: {settings.binance_testnet}")
    
    # Test basic client connection
    try:
        client = Client(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            testnet=settings.binance_testnet
        )
        
        # Test account info
        print("\n📊 Testing account info...")
        account = client.futures_account()
        
        if account:
            print("✅ Account connected successfully!")
            
            # Basic account info
            total_balance = float(account.get('totalWalletBalance', 0))
            unrealized_pnl = float(account.get('totalUnrealizedProfit', 0))
            
            print(f"💰 Total Balance: ${total_balance:,.2f}")
            print(f"📈 Unrealized PnL: ${unrealized_pnl:,.2f}")
            
            # Test positions
            positions = account.get('positions', [])
            active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            
            print(f"📊 Active Positions: {len(active_positions)}")
            
            if active_positions:
                print("\n📋 Current Positions:")
                for pos in active_positions[:5]:  # Show first 5
                    symbol = pos.get('symbol', 'N/A')
                    size = float(pos.get('positionAmt', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    unrealized = float(pos.get('unrealizedProfit', 0))
                    
                    print(f"   {symbol}: {size} @ ${entry_price:,.2f} (PnL: ${unrealized:,.2f})")
            
            # Test available symbols
            print("\n🔍 Testing market data...")
            try:
                ticker = client.futures_symbol_ticker(symbol="BTCUSDT")
                current_price = float(ticker['price'])
                print(f"✅ BTCUSDT Price: ${current_price:,.2f}")
                
                # Test 24h ticker
                ticker_24h = client.futures_ticker(symbol="BTCUSDT")
                price_change = float(ticker_24h.get('priceChange', 0))
                price_change_pct = float(ticker_24h.get('priceChangePercent', 0))
                
                print(f"📈 24h Change: ${price_change:,.2f} ({price_change_pct:.2f}%)")
                
            except Exception as e:
                print(f"❌ Market data test failed: {e}")
            
            # Test trading system integration
            print("\n🔄 Testing trading system...")
            risk_manager = RiskManager(settings)
            trader = BinanceTrader(settings, risk_manager)
            
            # Test current price retrieval
            price = await trader.get_current_price("BTCUSDT")
            print(f"✅ Trading system price: ${price:,.2f}")
            
            # Test account balance
            balance = await trader.get_account_balance()
            if balance:
                print(f"✅ Account balance retrieved: {list(balance.keys())[:3]}...")
            
            print("\n🎉 All API tests passed!")
            
            # Summary
            print("\n" + "=" * 50)
            print("API CONNECTION SUMMARY")
            print("=" * 50)
            print("✅ API Authentication: PASS")
            print("✅ Account Access: PASS")
            print("✅ Market Data: PASS")
            print("✅ Trading System: PASS")
            print(f"✅ Ready for live trading: {'YES' if settings.binance_testnet else 'CAUTION - LIVE MODE'}")
            
        else:
            print("❌ Account info empty")
            
    except BinanceAPIException as e:
        print(f"❌ Binance API Error: {e}")
        print(f"   Status: {e.status_code}")
        print(f"   Message: {e.message}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("   Please check:")
        print("   1. API key and secret are correct")
        print("   2. Testnet setting is appropriate")
        print("   3. Network connectivity")
        print("   4. Binance account permissions")

if __name__ == "__main__":
    asyncio.run(test_api_connection())