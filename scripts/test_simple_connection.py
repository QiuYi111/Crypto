#!/usr/bin/env python3
"""Simple API connection test."""

import os
from binance.client import Client
from src.cryptorl.config.settings import Settings

def simple_test():
    print("🔍 Testing API with current configuration...")
    
    settings = Settings()
    
    try:
        client = Client(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            testnet=settings.binance_testnet
        )
        
        # Simple ping test
        ping = client.ping()
        print("✅ Ping successful")
        
        # Test server time
        server_time = client.get_server_time()
        print(f"✅ Server time: {server_time}")
        
        # Test exchange info
        info = client.futures_exchange_info()
        print(f"✅ Exchange info: {len(info['symbols'])} symbols available")
        
        print("\n🎉 API connection successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 To fix:")
        if settings.binance_testnet:
            print("1. Go to https://testnet.binancefuture.com")
            print("2. Create new testnet account")
            print("3. Generate new API keys for testnet")
            print("4. Update .env file with new keys")
        else:
            print("1. Check your main Binance account API settings")
            print("2. Ensure Futures trading is enabled")
            print("3. Verify IP restrictions")

if __name__ == "__main__":
    simple_test()