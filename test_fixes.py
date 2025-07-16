#!/usr/bin/env python3
"""Simple test script to verify the missing methods are now available."""

import sys
from pathlib import Path
import inspect

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_market_data_collector():
    """Test MarketDataCollector has get_historical_data method."""
    try:
        from cryptorl.data.market_data import MarketDataCollector
        
        # Check if the method exists
        methods = [method for method in dir(MarketDataCollector) if not method.startswith('_')]
        
        if 'get_historical_data' in methods:
            print("✅ MarketDataCollector.get_historical_data method found")
            return True
        else:
            print("❌ MarketDataCollector.get_historical_data method NOT found")
            print("Available methods:", methods)
            return False
    except Exception as e:
        print(f"❌ Error testing MarketDataCollector: {e}")
        return False

def test_confidence_generator():
    """Test ConfidenceVectorGenerator has generate_for_period method."""
    try:
        from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator
        
        # Check if the method exists
        methods = [method for method in dir(ConfidenceVectorGenerator) if not method.startswith('_')]
        
        if 'generate_for_period' in methods:
            print("✅ ConfidenceVectorGenerator.generate_for_period method found")
            return True
        else:
            print("❌ ConfidenceVectorGenerator.generate_for_period method NOT found")
            print("Available methods:", methods)
            return False
    except Exception as e:
        print(f"❌ Error testing ConfidenceVectorGenerator: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing fixes for attribute access issues...")
    print("=" * 50)
    
    test1 = test_market_data_collector()
    test2 = test_confidence_generator()
    
    print("=" * 50)
    if test1 and test2:
        print("🎉 All fixes verified successfully!")
        print("The attribute access issues should now be resolved.")
    else:
        print("❌ Some fixes may still be needed.")
    
    return test1 and test2

if __name__ == "__main__":
    main()