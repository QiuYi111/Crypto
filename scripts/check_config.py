#!/usr/bin/env python3
"""Check Binance API configuration."""

import os
from src.cryptorl.config.settings import Settings

def check_config():
    print("🔍 Checking API Configuration...")
    print("=" * 40)
    
    settings = Settings()
    
    # Check all sources
    print("\n📋 Environment Variables:")
    print(f"   BINANCE_API_KEY: {'✅ Set' if os.getenv('BINANCE_API_KEY') else '❌ Missing'}")
    print(f"   BINANCE_SECRET_KEY: {'✅ Set' if os.getenv('BINANCE_SECRET_KEY') else '❌ Missing'}")
    print(f"   BINANCE_TESTNET: {'✅ Set' if os.getenv('BINANCE_TESTNET') else '❌ Missing'}")
    
    print("\n📋 Settings Values:")
    print(f"   api_key: {'✅ Loaded' if settings.binance_api_key else '❌ Missing'}")
    print(f"   secret_key: {'✅ Loaded' if settings.binance_secret_key else '❌ Missing'}")
    print(f"   testnet: {settings.binance_testnet}")
    
    print("\n📋 File Sources:")
    
    # Check .env file
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"   {env_file}: ✅ Found")
        with open(env_file) as f:
            content = f.read()
            print("   Contents (sanitized):")
            for line in content.split('\n'):
                if line.strip() and not line.startswith('#'):
                    key = line.split('=')[0]
                    print(f"     {key}=***")
    else:
        print(f"   {env_file}: ❌ Not found")
    
    # Check for secrets file
    secrets_file = ".env.local"
    if os.path.exists(secrets_file):
        print(f"   {secrets_file}: ✅ Found")
    else:
        print(f"   {secrets_file}: ❌ Not found")

if __name__ == "__main__":
    check_config()