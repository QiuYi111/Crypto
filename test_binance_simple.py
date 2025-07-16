#!/usr/bin/env python3
"""Simple test to verify Binance API connection and data retrieval."""

import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime

def test_binance_connection():
    """Test Binance API connection with public endpoints (no API key required)."""
    
    print("🧪 Testing Binance API Connection...")
    print("=" * 50)
    
    try:
        # Create client without API key for public endpoints
        client = Client()
        
        # Test 1: Get server time
        print("\n📅 Testing server time...")
        server_time = client.get_server_time()
        print(f"✅ Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        
        # Test 2: Get exchange info
        print("\n🏛️ Testing exchange info...")
        exchange_info = client.get_exchange_info()
        print(f"✅ Exchange: {exchange_info['exchangeFilters']}")
        
        # Test 3: Get BTC/USDT ticker
        print("\n💰 Testing BTC/USDT ticker...")
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        btc_price = float(ticker['price'])
        print(f"✅ BTC/USDT Price: ${btc_price:,.2f}")
        
        # Test 4: Get 24h statistics
        print("\n📊 Testing 24h statistics...")
        stats = client.get_ticker(symbol="BTCUSDT")
        price_change = float(stats['priceChange'])
        price_change_pct = float(stats['priceChangePercent'])
        volume = float(stats['volume'])
        
        print(f"📈 24h Change: ${price_change:,.2f} ({price_change_pct:.2f}%)")
        print(f"📊 24h Volume: {volume:,.2f} BTC")
        
        # Test 5: Get historical klines
        print("\n📈 Testing historical klines...")
        klines = client.get_historical_klines(
            symbol="BTCUSDT",
            interval=Client.KLINE_INTERVAL_1HOUR,
            start_str="1 day ago UTC"
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"✅ Retrieved {len(df)} hourly candles")
        print("\n📋 Last 5 candles:")
        print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail())
        
        # Test 6: Get multiple symbols
        print("\n🎯 Testing multiple symbols...")
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
        prices = {}
        
        for symbol in symbols:
            try:
                ticker = client.get_symbol_ticker(symbol=symbol)
                prices[symbol] = float(ticker['price'])
            except Exception as e:
                print(f"⚠️ Could not get {symbol}: {e}")
        
        print("✅ Current prices:")
        for symbol, price in prices.items():
            print(f"   {symbol}: ${price:,.4f}")
        
        print("\n🎉 All Binance API tests passed!")
        print("=" * 50)
        
        return True
        
    except BinanceAPIException as e:
        print(f"❌ Binance API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_binance_futures():
    """Test Binance futures endpoints."""
    
    print("\n🎯 Testing Binance Futures API...")
    print("=" * 50)
    
    try:
        client = Client()
        
        # Test futures exchange info
        print("\n🏛️ Testing futures exchange info...")
        futures_info = client.futures_exchange_info()
        usdt_pairs = [s for s in futures_info['symbols'] if s['quoteAsset'] == 'USDT']
        print(f"✅ Found {len(usdt_pairs)} USDT futures pairs")
        
        # Test futures ticker
        print("\n💰 Testing futures ticker...")
        futures_ticker = client.futures_symbol_ticker(symbol="BTCUSDT")
        btc_futures_price = float(futures_ticker['price'])
        print(f"✅ BTCUSDT Futures Price: ${btc_futures_price:,.2f}")
        
        # Test futures klines
        print("\n📈 Testing futures klines...")
        futures_klines = client.futures_historical_klines(
            symbol="BTCUSDT",
            interval=Client.KLINE_INTERVAL_1HOUR,
            start_str="1 day ago UTC"
        )
        print(f"✅ Retrieved {len(futures_klines)} futures hourly candles")
        
        print("\n🎉 Binance Futures API tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Futures API Error: {e}")
        return False

if __name__ == "__main__":
    # Test spot API
    spot_success = test_binance_connection()
    
    # Test futures API
    futures_success = test_binance_futures()
    
    if spot_success and futures_success:
        print("\n🚀 All tests completed successfully! Ready for trading data acquisition.")
    else:
        print("\n⚠️ Some tests failed. Please check your connection and try again.")