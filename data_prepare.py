#!/usr/bin/env python3
"""
Data Preparation Pipeline with Real LLM Integration

This script collects historical market data and generates real LLM confidence vectors
for BTC, ETH, and SOL. It prepares the complete dataset needed for training.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptorl.data.market_data import MarketDataCollector
from cryptorl.data.fusion import DataFusionEngine
from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator
from cryptorl.config.settings import Settings
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("logs/data_preparation.log", rotation="10 MB", level="INFO")

class DataPreparationPipeline:
    """Pipeline for preparing training data with real LLM integration."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.market_collector = MarketDataCollector(settings)
        self.fusion_engine = DataFusionEngine(settings)
        self.confidence_generator = ConfidenceVectorGenerator(settings)
        
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.data_dir = Path("data/training")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def collect_market_data(self, days: int = 90) -> dict:
        """Collect historical market data for all symbols."""
        logger.info(f"📊 Collecting {days} days of market data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        market_data = {}
        
        for symbol in self.symbols:
            logger.info(f"Fetching data for {symbol}...")
            
            try:
                # Collect hourly data
                data = await self.market_collector.get_historical_data(
                    symbol=symbol,
                    interval='1h',
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is not None and not data.empty:
                    market_data[symbol] = data
                    logger.info(f"✅ {symbol}: {len(data)} hourly records")
                    
                    # Save raw market data
                    data.to_csv(self.data_dir / f"{symbol}_market_data.csv")
                else:
                    logger.warning(f"⚠️ No data collected for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error collecting {symbol}: {e}")
        
        return market_data
    
    async def generate_confidence_vectors(self, market_data: dict) -> dict:
        """Generate real LLM confidence vectors for all symbols."""
        logger.info("🤖 Generating LLM confidence vectors...")
        
        confidence_data = {}
        
        for symbol, data in market_data.items():
            if data.empty:
                continue
                
            logger.info(f"Generating confidence vectors for {symbol}...")
            
            try:
                # Generate confidence vectors using real LLM
                confidence_vectors = await self.confidence_generator.generate_for_period(
                    symbol=symbol,
                    start_date=data.index.min(),
                    end_date=data.index.max(),
                    market_context=data
                )
                
                if confidence_vectors is not None:
                    confidence_data[symbol] = confidence_vectors
                    
                    # Save confidence vectors
                    confidence_vectors.to_csv(self.data_dir / f"{symbol}_confidence_vectors.csv")
                    logger.info(f"✅ {symbol}: {len(confidence_vectors)} confidence vectors")
                else:
                    logger.warning(f"⚠️ Failed to generate confidence for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error generating confidence for {symbol}: {e}")
        
        return confidence_data
    
    def fuse_datasets(self, market_data: dict, confidence_data: dict) -> dict:
        """Fuse market data with confidence vectors."""
        logger.info("🔗 Fusing datasets...")
        
        fused_datasets = {}
        
        for symbol in self.symbols:
            if symbol not in market_data or symbol not in confidence_data:
                logger.warning(f"⚠️ Skipping {symbol} - missing data")
                continue
            
            try:
                # Fuse the data
                fused = self.fusion_engine.fuse_data(
                    market_data=market_data[symbol],
                    confidence_vectors=confidence_data[symbol]
                )
                
                if not fused.empty:
                    fused_datasets[symbol] = fused
                    
                    # Save fused dataset
                    fused.to_csv(self.data_dir / f"{symbol}_fused_data.csv")
                    logger.info(f"✅ {symbol}: {len(fused)} fused records")
                    
                    # Log feature summary
                    logger.info(f"   Features: {list(fused.columns)}")
                    
            except Exception as e:
                logger.error(f"❌ Error fusing {symbol}: {e}")
        
        return fused_datasets
    
    def validate_data(self, fused_datasets: dict) -> bool:
        """Validate the prepared datasets."""
        logger.info("🔍 Validating datasets...")
        
        valid = True
        
        for symbol, data in fused_datasets.items():
            if data.empty:
                logger.error(f"❌ {symbol}: Empty dataset")
                valid = False
                continue
            
            # Check for NaN values
            nan_count = data.isnull().sum().sum()
            if nan_count > 0:
                logger.warning(f"⚠️ {symbol}: {nan_count} NaN values")
                # Forward fill missing values
                data = data.fillna(method='ffill')
                data = data.fillna(0)
            
            # Check date range
            date_range = data.index.max() - data.index.min()
            logger.info(f"✅ {symbol}: {len(data)} records, {date_range.days} days")
            
            # Save cleaned data
            data.to_csv(self.data_dir / f"{symbol}_cleaned.csv")
        
        return valid
    
    def create_training_splits(self, fused_datasets: dict):
        """Create training/validation/test splits."""
        logger.info("📊 Creating training splits...")
        
        splits_dir = self.data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        for symbol, data in fused_datasets.items():
            # Sort by date
            data = data.sort_index()
            
            # Split 70/15/15
            n = len(data)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            
            train_data = data[:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:]
            
            # Save splits
            train_data.to_csv(splits_dir / f"{symbol}_train.csv")
            val_data.to_csv(splits_dir / f"{symbol}_val.csv")
            test_data.to_csv(splits_dir / f"{symbol}_test.csv")
            
            logger.info(f"✅ {symbol}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    async def run(self, days: int = 90):
        """Run complete data preparation pipeline."""
        logger.info("🚀 Starting Data Preparation Pipeline...")
        
        try:
            # Step 1: Collect market data
            market_data = await self.collect_market_data(days)
            
            if not market_data:
                logger.error("❌ No market data collected")
                return False
            
            # Step 2: Generate confidence vectors
            confidence_data = await self.generate_confidence_vectors(market_data)
            
            if not confidence_data:
                logger.error("❌ No confidence vectors generated")
                return False
            
            # Step 3: Fuse datasets
            fused_datasets = self.fuse_datasets(market_data, confidence_data)
            
            if not fused_datasets:
                logger.error("❌ Failed to fuse datasets")
                return False
            
            # Step 4: Validate data
            if not self.validate_data(fused_datasets):
                logger.error("❌ Data validation failed")
                return False
            
            # Step 5: Create training splits
            self.create_training_splits(fused_datasets)
            
            logger.info("✅ Data preparation completed successfully!")
            
            # Print summary
            total_records = sum(len(data) for data in fused_datasets.values())
            logger.info(f"📊 Total records: {total_records}")
            logger.info(f"📈 Symbols: {list(fused_datasets.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return False

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data with LLM integration")
    parser.add_argument("--days", type=int, default=90, help="Number of days to collect")
    parser.add_argument("--symbols", nargs="+", default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    
    args = parser.parse_args()
    
    # Initialize settings
    settings = Settings()
    
    # Create pipeline
    pipeline = DataPreparationPipeline(settings)
    
    # Override symbols if provided
    pipeline.symbols = args.symbols
    
    # Run pipeline
    success = await pipeline.run(args.days)
    
    if success:
        print("\n🎉 Data preparation completed!")
        print("Next steps:")
        print("1. Run: python train.py")
        print("2. Run: python backtest.py")
    else:
        print("\n❌ Data preparation failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())