#!/usr/bin/env python3
"""Quick test for dataset generation."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptorl.config.settings import Settings
from cryptorl.data import InfluxDBClient, DataFusionEngine
from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator


async def test_basic_functionality():
    """Test basic dataset generation functionality."""
    
    logger.info("Testing basic dataset generation...")
    
    # Initialize components
    settings = Settings()
    influx_client = InfluxDBClient()
    
    try:
        # Test confidence generator
        logger.info("Testing confidence generator...")
        confidence_gen = ConfidenceVectorGenerator(settings, influx_client)
        
        # Test data fusion engine
        logger.info("Testing data fusion engine...")
        data_fusion = DataFusionEngine(settings, influx_client)
        
        # Test health checks
        logger.info("Running health checks...")
        confidence_health = await confidence_gen.health_check()
        fusion_health = await data_fusion.health_check()
        
        logger.info(f"Confidence generator health: {confidence_health}")
        logger.info(f"Data fusion health: {fusion_health}")
        
        # Test with minimal data
        symbols = ["BTCUSDT"]
        start_date = datetime.utcnow() - timedelta(days=3)
        end_date = datetime.utcnow() - timedelta(days=1)
        
        logger.info(f"Testing with symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Test data fusion
        dataset = await data_fusion.create_enhanced_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        logger.info(f"Generated dataset shape: {dataset.shape}")
        logger.info(f"Columns: {list(dataset.columns)}")
        
        if not dataset.empty:
            logger.info("Dataset sample:")
            logger.info(dataset.head())
            
            # Check confidence columns
            confidence_cols = [col for col in dataset.columns if col.startswith('confidence_')]
            logger.info(f"Confidence columns: {confidence_cols}")
            
            # Check for non-neutral values
            for col in confidence_cols:
                non_neutral = dataset[col] != 0.5
                if non_neutral.any():
                    logger.info(f"{col}: {non_neutral.sum()} non-neutral values")
                else:
                    logger.info(f"{col}: all neutral values (0.5)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False
    finally:
        influx_client.close()


async def main():
    """Main test function."""
    
    logger.add(
        "logs/test_dataset.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info("Starting dataset generation test...")
    
    success = await test_basic_functionality()
    
    if success:
        logger.info("✅ Dataset generation test completed successfully!")
    else:
        logger.error("❌ Dataset generation test failed!")


if __name__ == "__main__":
    asyncio.run(main())