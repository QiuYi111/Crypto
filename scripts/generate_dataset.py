#!/usr/bin/env python3
"""
Dataset generation script for CryptoRL agent.

This script creates enhanced datasets by combining historical market data 
with LLM-generated confidence vectors for RL training.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cryptorl.config.settings import Settings
from cryptorl.data import InfluxDBClient, DataFusionEngine
from cryptorl.llm.confidence_generator import ConfidenceVectorGenerator


class DatasetGenerator:
    """Enhanced dataset generation combining market data and confidence vectors."""
    
    def __init__(self):
        self.settings = Settings()
        self.influx_client = InfluxDBClient()
        self.data_fusion = DataFusionEngine(self.settings, self.influx_client)
        self.confidence_gen = ConfidenceVectorGenerator(self.settings, self.influx_client)
        
    async def generate_training_dataset(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        output_path: str = None
    ) -> pd.DataFrame:
        """Generate complete training dataset with market data and confidence vectors."""
        
        logger.info(f"Generating training dataset for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Interval: {interval}")
        
        try:
            # Create enhanced dataset
            dataset = await self.data_fusion.create_enhanced_dataset(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            if dataset.empty:
                logger.error("Generated dataset is empty!")
                return pd.DataFrame()
            
            logger.info(f"Generated dataset with {len(dataset)} rows and {len(dataset.columns)} columns")
            
            # Save dataset if path provided
            if output_path:
                await self._save_dataset(dataset, output_path)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            raise
    
    async def generate_confidence_only_dataset(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        output_path: str = None
    ) -> list:
        """Generate dataset with only confidence vectors for testing."""
        
        logger.info(f"Generating confidence vectors for {len(symbols)} symbols")
        
        vectors = await self.confidence_gen.batch_generate_historical(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            batch_size=5
        )
        
        logger.info(f"Generated {len(vectors)} confidence vectors")
        
        if output_path and vectors:
            await self._save_confidence_vectors(vectors, output_path)
        
        return vectors
    
    async def generate_sample_dataset(self, output_path: str = None) -> pd.DataFrame:
        """Generate a small sample dataset for testing."""
        
        logger.info("Generating sample dataset...")
        
        # Sample configuration
        symbols = ["BTCUSDT", "ETHUSDT"]
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow() - timedelta(days=1)
        
        dataset = await self.generate_training_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            output_path=output_path
        )
        
        return dataset
    
    async def validate_dataset(self, dataset: pd.DataFrame) -> dict:
        """Validate the generated dataset for completeness."""
        
        validation_results = {
            "total_rows": len(dataset),
            "symbols": dataset['symbol'].unique().tolist() if 'symbol' in dataset.columns else [],
            "date_range": None,
            "missing_values": {},
            "confidence_coverage": {}
        }
        
        if not dataset.empty:
            # Check date range
            if 'date' in dataset.columns:
                dates = pd.to_datetime(dataset['date'])
                validation_results["date_range"] = {
                    "start": dates.min().strftime('%Y-%m-%d'),
                    "end": dates.max().strftime('%Y-%m-%d')
                }
            
            # Check for missing values
            missing_counts = dataset.isnull().sum()
            validation_results["missing_values"] = {
                col: int(count) for col, count in missing_counts[missing_counts > 0].items()
            }
            
            # Check confidence vector coverage
            confidence_cols = [col for col in dataset.columns if col.startswith('confidence_')]
            for col in confidence_cols:
                non_neutral = dataset[col] != 0.5  # Assuming 0.5 is neutral
                validation_results["confidence_coverage"][col] = {
                    "total_values": len(dataset),
                    "non_neutral_values": int(non_neutral.sum()),
                    "coverage_rate": float(non_neutral.sum() / len(dataset))
                }
        
        return validation_results
    
    async def _save_dataset(self, dataset: pd.DataFrame, output_path: str):
        """Save dataset to file."""
        
        try:
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
                "symbols": dataset['symbol'].unique().tolist() if 'symbol' in dataset.columns else [],
                "date_range": {
                    "start": dataset['date'].min() if 'date' in dataset.columns else None,
                    "end": dataset['date'].max() if 'date' in dataset.columns else None
                }
            }
            
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    async def _save_confidence_vectors(self, vectors: list, output_path: str):
        """Save confidence vectors to file."""
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'date': v.date,
                    'symbol': v.symbol,
                    'fundamentals': v.fundamentals,
                    'industry_condition': v.industry_condition,
                    'geopolitics': v.geopolitics,
                    'macroeconomics': v.macroeconomics,
                    'technical_sentiment': v.technical_sentiment,
                    'regulatory_impact': v.regulatory_impact,
                    'innovation_impact': v.innovation_impact,
                    'confidence_score': v.confidence_score,
                    'reasoning': v.reasoning
                }
                for v in vectors
            ])
            
            # Save as CSV
            csv_path = output_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Confidence vectors saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving confidence vectors: {e}")
            raise
    
    def close(self):
        """Clean up resources."""
        self.influx_client.close()


async def main():
    """Main function for dataset generation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="CryptoRL Dataset Generator")
    parser.add_argument(
        "--mode",
        choices=["full", "sample", "confidence"],
        default="sample",
        help="Generation mode"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Symbols to include"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to collect (if start-date not provided)"
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Data interval (1d, 4h, 1h)"
    )
    parser.add_argument(
        "--output",
        default="data/enhanced_dataset",
        help="Output path prefix"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated dataset"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.add(
        "logs/dataset_generation.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = datetime.utcnow() - timedelta(days=args.days)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.utcnow() - timedelta(days=1)
    
    logger.info("Starting CryptoRL dataset generation")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Interval: {args.interval}")
    
    generator = DatasetGenerator()
    
    try:
        if args.mode == "sample":
            dataset = await generator.generate_sample_dataset(args.output)
        elif args.mode == "confidence":
            vectors = await generator.generate_confidence_only_dataset(
                symbols=args.symbols,
                start_date=start_date,
                end_date=end_date,
                output_path=args.output
            )
            dataset = pd.DataFrame()  # Empty for confidence-only mode
        else:  # full
            dataset = await generator.generate_training_dataset(
                symbols=args.symbols,
                start_date=start_date,
                end_date=end_date,
                interval=args.interval,
                output_path=args.output
            )
        
        # Validate dataset
        if args.validate and not dataset.empty:
            validation = await generator.validate_dataset(dataset)
            logger.info("Dataset validation results:")
            logger.info(json.dumps(validation, indent=2))
            
            # Save validation results
            validation_path = Path(args.output).with_suffix('.validation.json')
            with open(validation_path, 'w') as f:
                json.dump(validation, f, indent=2)
            logger.info(f"Validation results saved to {validation_path}")
        
        logger.info("Dataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        raise
    finally:
        generator.close()


if __name__ == "__main__":
    asyncio.run(main())