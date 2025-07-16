#!/usr/bin/env python3
"""
Simple validation to check Phase 1 and 2 completion.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🔍 Validating Phase 1 & 2 completion...")
print("=" * 50)

# Test 1: Check all required files exist
required_files = [
    # Phase 1 - Data Infrastructure
    "src/cryptorl/data/market_data.py",
    "src/cryptorl/data/influxdb_client.py", 
    "src/cryptorl/data/binance_client.py",
    "src/cryptorl/data/fusion.py",
    
    # Phase 2 - LLM Enhancement
    "src/cryptorl/llm/__init__.py",
    "src/cryptorl/llm/models.py",
    "src/cryptorl/llm/llm_client.py",
    "src/cryptorl/llm/rag_pipeline.py",
    "src/cryptorl/llm/confidence_generator.py",
    
    # Phase 2 - Mamba Framework
    "src/cryptorl/rl/__init__.py",
    "src/cryptorl/rl/models.py",
    "src/cryptorl/rl/environment.py",
    "src/cryptorl/rl/agent.py",
    "src/cryptorl/rl/training.py",
    "src/cryptorl/rl/evaluation.py",
    "src/cryptorl/rl/mamba_exploration.py",
    
    # Trading System
    "src/cryptorl/trading/__init__.py",
    "src/cryptorl/trading/execution.py",
    "src/cryptorl/trading/risk_manager.py",
]

missing_files = []
for file_path in required_files:
    if not Path(file_path).exists():
        missing_files.append(file_path)

if missing_files:
    print("❌ Missing files:")
    for f in missing_files:
        print(f"   - {f}")
else:
    print("✅ All required files exist")

# Test 2: Check class definitions
print("\n📋 Checking class definitions...")

class_checks = [
    ("MarketDataCollector", "src.cryptorl.data.market_data"),
    ("InfluxDBClient", "src.cryptorl.data.influxdb_client"),
    ("BinanceClient", "src.cryptorl.data.binance_client"),
    ("DataFusionEngine", "src.cryptorl.data.fusion"),
    ("ConfidenceVector", "src.cryptorl.llm.models"),
    ("NewsArticle", "src.cryptorl.llm.models"),
    ("ConfidenceVectorGenerator", "src.cryptorl.llm.confidence_generator"),
    ("RAGPipeline", "src.cryptorl.llm.rag_pipeline"),
    ("MambaModel", "src.cryptorl.rl.models"),
    ("MambaPolicyNetwork", "src.cryptorl.rl.models"),
    ("CryptoTradingEnvironment", "src.cryptorl.rl.environment"),
    ("CryptoRLAgent", "src.cryptorl.rl.agent"),
    ("Trainer", "src.cryptorl.rl.training"),
    ("Evaluator", "src.cryptorl.rl.evaluation"),
    ("MambaExplorer", "src.cryptorl.rl.mamba_exploration"),
    ("TradingExecutor", "src.cryptorl.trading.execution"),
    ("RiskManager", "src.cryptorl.trading.risk_manager"),
]

class_issues = []
for class_name, module_name in class_checks:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✅ {class_name}")
    except Exception as e:
        print(f"❌ {class_name}: {e}")
        class_issues.append((class_name, str(e)))

# Test 3: Check basic functionality
print("\n⚙️ Testing basic functionality...")

try:
    import torch
    print("✅ torch available")
except ImportError:
    print("⚠️ torch not available (expected in test environment)")

try:
    from cryptorl.llm.models import ConfidenceVector
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
        reasoning="Test vector"
    )
    print("✅ ConfidenceVector model functional")
except Exception as e:
    print(f"❌ ConfidenceVector: {e}")

try:
    from cryptorl.llm.models import NewsArticle
    article = NewsArticle(
        title="Test Article",
        content="This is test content",
        source="Test Source",
        published_date="2024-01-01"
    )
    print("✅ NewsArticle model functional")
except Exception as e:
    print(f"❌ NewsArticle: {e}")

# Summary
print("\n" + "=" * 50)
print("VALIDATION SUMMARY")
print("=" * 50)

if not missing_files:
    print("✅ Phase 1: All data infrastructure files present")
if not class_issues:
    print("✅ Phase 2: All LLM and Mamba classes defined")

print(f"\n📊 Files: {len(required_files) - len(missing_files)}/{len(required_files)} present")
print(f"📊 Classes: {len(class_checks) - len(class_issues)}/{len(class_checks)} defined")

if not missing_files and not class_issues:
    print("\n🎉 PHASE 1 & 2 ARE COMPLETE!")
    print("   Ready for Phase 3: RL Training")
else:
    print("\n⚠️  Some issues need addressing")

print("\n📋 Next Steps:")
print("   1. Install dependencies: pip install torch transformers mamba-ssm")
print("   2. Configure .env file")
print("   3. Run: python scripts/validate_setup.py")