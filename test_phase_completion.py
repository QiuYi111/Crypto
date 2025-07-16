#!/usr/bin/env python3
"""
Comprehensive test script for Phase 1 & 2 completion validation.
"""

import sys
import os
from pathlib import Path

def test_phase_1():
    """Test Phase 1: Data Infrastructure"""
    print("🔍 Testing Phase 1: Data Infrastructure...")
    
    tests = []
    
    try:
        from cryptorl.data.market_data import MarketDataCollector
        tests.append(("MarketDataCollector", True))
    except Exception as e:
        tests.append(("MarketDataCollector", False, str(e)))
    
    try:
        from cryptorl.data.influxdb_client import InfluxDBClient
        tests.append(("InfluxDBClient", True))
    except Exception as e:
        tests.append(("InfluxDBClient", False, str(e)))
    
    try:
        from cryptorl.data.binance_client import BinanceClient
        tests.append(("BinanceClient", True))
    except Exception as e:
        tests.append(("BinanceClient", False, str(e)))
    
    phase1_passed = sum(1 for _, result, *_ in tests if result)
    print(f"Phase 1: {phase1_passed}/{len(tests)} components working")
    
    for name, result, *error in tests:
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
        if error:
            print(f"     Error: {error[0]}")
    
    return phase1_passed == len(tests)

def test_phase_2():
    """Test Phase 2: LLM Enhancement & Mamba Framework"""
    print("\n🔍 Testing Phase 2: LLM Enhancement & Mamba Framework...")
    
    tests = []
    
    try:
        from cryptorl.llm.models import ConfidenceVector, NewsArticle, SearchQuery
        tests.append(("LLM Models", True))
    except Exception as e:
        tests.append(("LLM Models", False, str(e)))
    
    try:
        from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator
        tests.append(("ConfidenceGenerator", True))
    except Exception as e:
        tests.append(("ConfidenceGenerator", False, str(e)))
    
    try:
        from cryptorl.llm.rag_pipeline import RAGPipeline
        tests.append(("RAGPipeline", True))
    except Exception as e:
        tests.append(("RAGPipeline", False, str(e)))
    
    try:
        from cryptorl.rl.models import MambaModel, MambaPolicyNetwork
        tests.append(("Mamba Models", True))
    except Exception as e:
        tests.append(("Mamba Models", False, str(e)))
    
    try:
        from cryptorl.rl.environment import CryptoTradingEnvironment
        tests.append(("RL Environment", True))
    except Exception as e:
        tests.append(("RL Environment", False, str(e)))
    
    try:
        from cryptorl.data.fusion import DataFusionEngine
        tests.append(("DataFusionEngine", True))
    except Exception as e:
        tests.append(("DataFusionEngine", False, str(e)))
    
    phase2_passed = sum(1 for _, result, *_ in tests if result)
    print(f"Phase 2: {phase2_passed}/{len(tests)} components working")
    
    for name, result, *error in tests:
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
        if error:
            print(f"     Error: {error[0]}")
    
    return phase2_passed == len(tests)

def test_model_functionality():
    """Test basic model functionality"""
    print("\n🔍 Testing Model Functionality...")
    
    try:
        import torch
        from cryptorl.rl.models import create_mamba_model
        
        # Test Mamba model creation
        model_config = {
            'hidden_dim': 64,
            'num_layers': 2,
            'use_mamba': True,
            'continuous_actions': True
        }
        
        model = create_mamba_model(
            observation_space=10,
            action_space=1,
            model_config=model_config
        )
        
        # Test forward pass
        dummy_input = torch.randn(1, 30, 10)
        output = model(dummy_input)
        
        print("✅ Mamba model functional")
        if isinstance(output, tuple):
            print(f"   Output type: {type(output)} with {len(output)} elements")
        else:
            print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model functionality test failed: {e}")
        return False

def test_llm_models():
    """Test LLM model creation"""
    print("\n🔍 Testing LLM Models...")
    
    try:
        from cryptorl.llm.models import ConfidenceVector, NewsArticle
        
        # Test confidence vector creation
        cv = ConfidenceVector(
            symbol="BTCUSDT",
            date="2024-01-01",
            fundamentals=0.8,
            industry_condition=0.7,
            geopolitics=0.6,
            macroeconomics=0.75,
            technical_sentiment=0.8,
            regulatory_impact=0.7,
            innovation_impact=0.9,
            confidence_score=0.8,
            reasoning="Test confidence vector"
        )
        
        print("✅ LLM models functional")
        print(f"   Confidence vector: {cv.to_array()}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM models test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CryptoRL Phase 1 & 2 Completion Test")
    print("=" * 50)
    
    phase1_complete = test_phase_1()
    phase2_complete = test_phase_2()
    
    # Test functionality
    model_func = test_model_functionality()
    llm_func = test_llm_models()
    
    print("\n" + "=" * 50)
    print("COMPLETION SUMMARY")
    print("=" * 50)
    print(f"Phase 1 (Data Infrastructure): {'✅ COMPLETE' if phase1_complete else '❌ INCOMPLETE'}")
    print(f"Phase 2 (LLM Enhancement): {'✅ COMPLETE' if phase2_complete else '❌ INCOMPLETE'}")
    print(f"Model Functionality: {'✅ WORKING' if model_func else '❌ ISSUES'}")
    print(f"LLM Functionality: {'✅ WORKING' if llm_func else '❌ ISSUES'}")
    
    all_complete = phase1_complete and phase2_complete and model_func and llm_func
    
    if all_complete:
        print("\n🎉 Both Phase 1 and Phase 2 are COMPLETE!")
        print("   Ready to proceed to Phase 3: RL Training")
    else:
        print("\n⚠️  Some components need attention")
    
    return all_complete

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)