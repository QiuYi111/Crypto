#!/usr/bin/env python3
"""Live test with real Binance API data."""

import asyncio
from src.cryptorl.config.settings import Settings
from src.cryptorl.trading.execution import BinanceTrader
from src.cryptorl.risk_management.risk_manager import RiskManager
from src.cryptorl.trading.execution import Order

async def live_test():
    print("🚀 Starting Live API Test...")
    print("=" * 50)
    
    settings = Settings()
    risk_manager = RiskManager(settings)
    trader = BinanceTrader(settings, risk_manager)
    
    # Test 1: Account Balance
    print("\n💰 Account Balance:")
    try:
        balance = await trader.get_account_balance()
        print(f"   Assets: {list(balance.keys())}")
        if 'USDT' in balance:
            print(f"   USDT Balance: ${balance['USDT']:,.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Current Prices
    print("\n📈 Current Prices:")
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    for symbol in symbols:
        try:
            price = await trader.get_current_price(symbol)
            print(f"   {symbol}: ${price:,.2f}")
        except Exception as e:
            print(f"   {symbol}: Error - {e}")
    
    # Test 3: Positions
    print("\n📊 Current Positions:")
    for symbol in symbols:
        try:
            position = await trader.get_position(symbol)
            if position:
                print(f"   {symbol}: {position.quantity} @ ${position.entry_price:,.2f} (PnL: ${position.unrealized_pnl:,.2f})")
            else:
                print(f"   {symbol}: No position")
        except Exception as e:
            print(f"   {symbol}: Error - {e}")
    
    # Test 4: Risk Evaluation
    print("\n🛡️  Risk Analysis:")
    try:
        for symbol in symbols[:1]:  # Test BTC only
            price = await trader.get_current_price(symbol)
            risk_metrics = risk_manager.evaluate_rymbol(symbol, price, 0.1)  # Small position
            print(f"   {symbol} 0.1 position: Risk Level {risk_metrics.risk_level.value}")
    except Exception as e:
        print(f"   Risk eval error: {e}")
    
    print("\n✅ Live test complete!")

if __name__ == "__main__":
    asyncio.run(live_test())