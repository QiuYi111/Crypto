#!/usr/bin/env python3
"""
Data collection script for CryptoRL agent.

This script demonstrates how to collect historical market data from Binance
and store it in InfluxDB.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cryptorl.data import BinanceClient, MarketDataCollector, InfluxDBClient
from loguru import logger


async def collect_historical_data():
    """Collect historical data for major crypto pairs."""
    
    # Initialize clients
    influx_client = InfluxDBClient()
    binance_client = BinanceClient(testnet=True)
    collector = MarketDataCollector(binance_client, influx_client)
    
    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = ["1h", "4h", "1d"]
    start_date = datetime.utcnow() - timedelta(days=30)  # Last 30 days
    
    logger.info(f"Starting data collection for symbols: {symbols}")
    logger.info(f"Intervals: {intervals}")
    logger.info(f"Start date: {start_date}")
    
    try:
        results = await collector.collect_all_historical_data(
            symbols=symbols,
            intervals=intervals,
            start_date=start_date
        )
        
        logger.info("Data collection completed!")
        
        for symbol, symbol_results in results.items():
            logger.info(f"\n{symbol}:")
            for interval, count in symbol_results.items():
                logger.info(f"  {interval}: {count} records")
                
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        raise
    finally:
        influx_client.close()


async def collect_real_time_demo():
    """Demo real-time data collection (runs for 60 seconds)."""
    
    influx_client = InfluxDBClient()
    binance_client = BinanceClient(testnet=True)
    collector = MarketDataCollector(binance_client, influx_client)
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    logger.info(f"Starting real-time data collection demo for {symbols}")
    logger.info("This will run for 60 seconds...")
    
    try:
        # Run for 60 seconds then stop
        task = asyncio.create_task(
            collector.collect_real_time_data(symbols)
        )
        
        await asyncio.sleep(60)
        task.cancel()
        
        logger.info("Real-time demo completed")
        
    except asyncio.CancelledError:
        logger.info("Real-time collection stopped")
    except Exception as e:
        logger.error(f"Error in real-time demo: {e}")
    finally:
        influx_client.close()


async def main():
    """Main function to run data collection."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="CryptoRL Data Collection")
    parser.add_argument(
        "--mode", 
        choices=["historical", "realtime"], 
        default="historical",
        help="Collection mode"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Symbols to collect"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to collect (for historical mode)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.add(
        "logs/data_collection.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info("Starting CryptoRL data collection")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {args.symbols}")
    
    if args.mode == "historical":
        await collect_historical_data()
    elif args.mode == "realtime":
        await collect_real_time_demo()
    
    logger.info("Data collection script completed")


if __name__ == "__main__":
    asyncio.run(main())