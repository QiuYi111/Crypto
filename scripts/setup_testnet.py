#!/usr/bin/env python3
"""Setup guide for Binance testnet API keys."""

import os
from pathlib import Path

def setup_testnet_guide():
    print("🎯 Binance Testnet Setup Guide")
    print("=" * 50)
    
    print("\n📋 Current Status:")
    print("   ✅ API connectivity working")
    print("   ❌ Testnet API keys invalid")
    
    print("\n🔧 To use testnet, you need to:")
    print("   1. Go to: https://testnet.binancefuture.com")
    print("   2. Create a testnet account (it's free)")
    print("   3. Generate new API keys specifically for testnet")
    print("   4. Update your .env file with new keys")
    
    print("\n📝 Steps:")
    print("   1. Visit https://testnet.binancefuture.com")
    print("   2. Click 'Login' and create account")
    print("   3. Go to API Management")
    print("   4. Create new API key")
    print("   5. Copy the API Key and Secret")
    print("   6. Update your .env file:")
    print("      BINANCE_API_KEY=your_new_testnet_key")
    print("      BINANCE_SECRET_KEY=your_new_testnet_secret")
    print("      BINANCE_TESTNET=true")
    
    print("\n🔄 Alternative: Switch to mainnet")
    print("   If you want to use mainnet instead:")
    print("   1. Change BINANCE_TESTNET=false in .env")
    print("   2. Ensure your mainnet API keys have futures permissions")
    print("   3. Ensure IP restrictions allow your current IP")

if __name__ == "__main__":
    setup_testnet_guide()