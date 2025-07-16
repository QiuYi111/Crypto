#!/usr/bin/env python3
"""
Mock dataset generation script for testing CryptoRL agent.

This script creates mock datasets with realistic market data and 
simulated confidence vectors for development and testing.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptorl.config.settings import Settings
from cryptorl.data import InfluxDBClient


class MockDatasetGenerator:
    """Generate mock datasets for testing and development."""
    
    def __init__(self):
        self.settings = Settings()
        
    def generate_mock_market_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Generate realistic mock market data with OHLCV."""
        
        logger.info(f"Generating mock market data for {len(symbols)} symbols")
        
        # Determine frequency based on interval
        freq_map = {
            "1d": "D",
            "4h": "4H",
            "1h": "H",
            "30m": "30T",
            "15m": "15T"
        }
        
        freq = freq_map.get(interval, "D")
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        all_data = []
        
        for symbol in symbols:
            # Base price and volatility for each symbol
            base_prices = {
                "BTCUSDT": 45000,
                "ETHUSDT": 3000,
                "SOLUSDT": 100,
                "ADAUSDT": 0.5,
                "DOTUSDT": 8
            }
            
            base_volatilities = {
                "BTCUSDT": 0.03,
                "ETHUSDT": 0.04,
                "SOLUSDT": 0.06,
                "ADAUSDT": 0.05,
                "DOTUSDT": 0.045
            }
            
            base_price = base_prices.get(symbol, 1000)
            volatility = base_volatilities.get(symbol, 0.04)
            
            # Generate price data with realistic patterns
            n_periods = len(date_range)
            returns = np.random.normal(0, volatility, n_periods)
            
            # Add some trend and momentum
            trend = np.linspace(0, 0.1, n_periods)  # Slight upward trend
            momentum = np.convolve(returns, [0.2, 0.3, 0.3, 0.2], mode='same')
            
            cumulative_returns = np.cumsum(returns + trend + momentum * 0.3)
            prices = base_price * (1 + cumulative_returns)
            
            # Generate OHLCV data
            for i, (date, price) in enumerate(zip(date_range, prices)):
                daily_vol = volatility * (1 + 0.5 * np.random.random())
                
                # Generate OHLC from close price
                close = max(price, base_price * 0.1)  # Prevent negative prices
                
                # Generate high, low, open around close
                high = close * (1 + abs(np.random.normal(0, daily_vol)))
                low = close * (1 - abs(np.random.normal(0, daily_vol)))
                open_price = close * (1 + np.random.normal(0, daily_vol * 0.5))
                
                # Ensure logical ordering
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                # Generate volume (correlated with volatility)
                base_volume = {
                    "BTCUSDT": 1000000,
                    "ETHUSDT": 500000,
                    "SOLUSDT": 200000,
                    "ADAUSDT": 1000000,
                    "DOTUSDT": 300000
                }
                
                volume = base_volume.get(symbol, 100000) * (1 + abs(returns[i])) * np.random.uniform(0.5, 2)
                
                all_data.append({
                    'date': date,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'quote_volume': volume * close,
                    'trades': int(volume / np.random.uniform(100, 1000))
                })
        
        df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(df)} mock market data records")
        return df
    
    def generate_mock_confidence_vectors(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate mock confidence vectors with realistic patterns."""
        
        logger.info(f"Generating mock confidence vectors for {len(symbols)} symbols")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        all_vectors = []
        
        for symbol in symbols:
            for date in date_range:
                # Generate confidence vectors with some correlation to market conditions
                market_sentiment = np.random.normal(0.5, 0.2)
                market_sentiment = np.clip(market_sentiment, 0, 1)
                
                # Different dimensions with some correlation
                fundamentals = np.clip(np.random.beta(2, 2), 0.3, 0.8)
                industry = np.clip(market_sentiment + np.random.normal(0, 0.1), 0.2, 0.9)
                geopolitics = np.clip(0.5 + np.random.normal(0, 0.15), 0.1, 0.9)
                macroeconomics = np.clip(0.5 + np.random.normal(0, 0.1), 0.3, 0.8)
                technical = np.clip(market_sentiment + np.random.normal(0, 0.2), 0.2, 0.9)
                regulatory = np.clip(np.random.beta(1.5, 2.5), 0.1, 0.9)
                innovation = np.clip(np.random.beta(2, 1.5), 0.3, 0.9)
                
                all_vectors.append({
                    'date': date,
                    'symbol': symbol,
                    'confidence_fundamentals': fundamentals,
                    'confidence_industry': industry,
                    'confidence_geopolitics': geopolitics,
                    'confidence_macro': macroeconomics,
                    'confidence_technical': technical,
                    'confidence_regulatory': regulatory,
                    'confidence_innovation': innovation,
                    'confidence_overall': np.mean([fundamentals, industry, geopolitics, macroeconomics])
                })
        
        df = pd.DataFrame(all_vectors)
        logger.info(f"Generated {len(df)} mock confidence vectors")
        return df
    
    def generate_enhanced_dataset(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Generate complete enhanced dataset with market data and confidence vectors."""
        
        logger.info("Generating enhanced mock dataset...")
        
        # Generate market data
        market_data = self.generate_mock_market_data(symbols, start_date, end_date, interval)
        
        # Generate confidence vectors
        confidence_data = self.generate_mock_confidence_vectors(symbols, start_date, end_date)
        
        # Merge datasets
        enhanced_data = pd.merge(
            market_data,
            confidence_data,
            on=['date', 'symbol'],
            how='left'
        )
        
        # Add technical indicators
        enhanced_data = self._add_technical_indicators(enhanced_data)
        
        # Add asset encoding
        enhanced_data = self._add_asset_encoding(enhanced_data, symbols)
        
        logger.info(f"Generated enhanced dataset with {len(enhanced_data)} rows")
        return enhanced_data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        
        # Price changes
        df['price_change'] = df.groupby('symbol')['close'].pct_change()
        
        # Moving averages
        df['sma_7'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df['sma_30'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        df['sma_ratio'] = df['sma_7'] / df['sma_30']
        
        # RSI calculation
        def calculate_rsi(prices, period=14):
            if len(prices) < period:
                return np.nan
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi'] = df.groupby('symbol')['close'].transform(calculate_rsi)
        
        # Volume indicators
        df['volume_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        df['volatility'] = df.groupby('symbol')['price_change'].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )
        
        # Price position in range
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _add_asset_encoding(self, df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
        """Add one-hot encoding for different assets."""
        
        # Create one-hot encoding
        for symbol in symbols:
            df[f'asset_{symbol}'] = (df['symbol'] == symbol).astype(int)
        
        # Also add asset ID for embedding
        symbol_to_id = {symbol: i for i, symbol in enumerate(symbols)}
        df['asset_id'] = df['symbol'].map(symbol_to_id)
        
        return df
    
    def save_dataset(self, dataset: pd.DataFrame, output_path: str):
        """Save the dataset to files."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_path.with_suffix('.csv')
        dataset.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved to {csv_path}")
        
        # Save as Parquet for better compression
        parquet_path = output_path.with_suffix('.parquet')
        dataset.to_parquet(parquet_path, index=False)
        logger.info(f"Dataset saved to {parquet_path}")
        
        # Save metadata
        metadata = {
            "generation_time": datetime.utcnow().isoformat(),
            "shape": [len(dataset), len(dataset.columns)],
            "columns": dataset.columns.tolist(),
            "symbols": dataset['symbol'].unique().tolist(),
            "date_range": {
                "start": dataset['date'].min().isoformat(),
                "end": dataset['date'].max().isoformat()
            },
            "null_counts": dataset.isnull().sum().to_dict()
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_path}")


async def main():
    """Main function for mock dataset generation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="CryptoRL Mock Dataset Generator")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Symbols to include"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to generate"
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Data interval (1d, 4h, 1h)"
    )
    parser.add_argument(
        "--output",
        default="data/mock_enhanced_dataset",
        help="Output path prefix"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.add(
        "logs/mock_dataset_generation.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Set dates
    end_date = datetime.utcnow() - timedelta(days=1)
    start_date = end_date - timedelta(days=args.days)
    
    logger.info("Starting mock dataset generation")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Interval: {args.interval}")
    
    generator = MockDatasetGenerator()
    
    try:
        # Generate enhanced dataset
        dataset = generator.generate_enhanced_dataset(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval
        )
        
        # Save dataset
        generator.save_dataset(dataset, args.output)
        
        # Print summary
        logger.info("Dataset generation completed!")
        logger.info(f"Dataset shape: {dataset.shape}")
        logger.info(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
        logger.info(f"Symbols: {dataset['symbol'].unique().tolist()}")
        
        # Show sample
        logger.info("\nDataset sample:")
        logger.info(dataset[['date', 'symbol', 'close', 'volume', 'confidence_overall']].head())
        
    except Exception as e:
        logger.error(f"Error during mock dataset generation: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())