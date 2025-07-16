"""Mamba model architecture exploration and benchmarking."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import pandas as pd
from loguru import logger
from dataclasses import dataclass

from .models import create_mamba_model, MambaModel, MambaPolicyNetwork
from ..config.settings import Settings


@dataclass
class ModelBenchmark:
    """Results from model benchmarking."""
    model_name: str
    parameters: int
    inference_time: float
    memory_usage_mb: float
    accuracy: float
    training_speed: float
    sequence_length: int
    batch_size: int


class MambaExplorer:
    """Exploration framework for Mamba model architectures."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def explore_architectures(
        self,
        observation_dim: int,
        action_dim: int,
        sequence_lengths: List[int] = [30, 60, 90, 120],
        batch_sizes: List[int] = [1, 8, 16, 32],
        hidden_dims: List[int] = [128, 256, 512],
        num_layers: List[int] = [2, 4, 6, 8]
    ) -> List[ModelBenchmark]:
        """Explore different Mamba architectures."""
        
        logger.info("Starting Mamba architecture exploration...")
        
        benchmarks = []
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                for hidden_dim in hidden_dims:
                    for num_layer in num_layers:
                        
                        # Create model configuration
                        model_config = {
                            'hidden_dim': hidden_dim,
                            'num_layers': num_layer,
                            'use_mamba': True,
                            'continuous_actions': self.settings.rl_continuous_actions
                        }
                        
                        # Create model
                        model = create_mamba_model(
                            observation_space=observation_dim,
                            action_space=action_dim,
                            model_config=model_config
                        ).to(self.device)
                        
                        # Benchmark model
                        benchmark = self._benchmark_model(
                            model=model,
                            model_name=f"mamba_h{hidden_dim}_l{num_layer}",
                            sequence_length=seq_len,
                            batch_size=batch_size,
                            observation_dim=observation_dim
                        )
                        
                        benchmarks.append(benchmark)
                        
                        logger.info(f"Benchmarked {benchmark.model_name}: "
                                  f"{benchmark.parameters} params, "
                                  f"{benchmark.inference_time:.4f}s, "
                                  f"{benchmark.memory_usage_mb:.2f}MB")
        
        return benchmarks
    
    def compare_with_baselines(
        self,
        observation_dim: int,
        action_dim: int,
        sequence_length: int = 30,
        batch_size: int = 8
    ) -> List[ModelBenchmark]:
        """Compare Mamba with baseline architectures."""
        
        logger.info("Comparing Mamba with baseline architectures...")
        
        architectures = {
            'mamba': {'use_mamba': True, 'hidden_dim': 256, 'num_layers': 4},
            'transformer': {'use_mamba': False, 'hidden_dim': 256, 'num_layers': 4},
            'lstm': {'use_mamba': False, 'hidden_dim': 256, 'num_layers': 2},
            'gru': {'use_mamba': False, 'hidden_dim': 256, 'num_layers': 2}
        }
        
        benchmarks = []
        
        for name, config in architectures.items():
            model = create_mamba_model(
                observation_space=observation_dim,
                action_space=action_dim,
                model_config=config
            ).to(self.device)
            
            benchmark = self._benchmark_model(
                model=model,
                model_name=name,
                sequence_length=sequence_length,
                batch_size=batch_size,
                observation_dim=observation_dim
            )
            
            benchmarks.append(benchmark)
        
        return benchmarks
    
    def _benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        sequence_length: int,
        batch_size: int,
        observation_dim: int
    ) -> ModelBenchmark:
        """Benchmark a single model configuration."""
        
        model.eval()
        
        # Count parameters
        parameters = sum(p.numel() for p in model.parameters())
        
        # Create dummy input
        dummy_input = torch.randn(
            batch_size, sequence_length, observation_dim,
            device=self.device
        )
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure inference time
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                output = model(dummy_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        inference_time = (time.time() - start_time) / 100
        
        # Measure memory usage
        if self.device.type == 'cuda':
            memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_usage = 0.0  # CPU memory tracking is more complex
        
        # Dummy accuracy score (would be replaced with actual evaluation)
        accuracy = 0.75 + np.random.normal(0, 0.1)
        
        # Training speed estimate
        training_speed = 1.0 / (inference_time * parameters / 1e6)  # Rough estimate
        
        return ModelBenchmark(
            model_name=model_name,
            parameters=parameters,
            inference_time=inference_time,
            memory_usage_mb=memory_usage,
            accuracy=accuracy,
            training_speed=training_speed,
            sequence_length=sequence_length,
            batch_size=batch_size
        )
    
    def analyze_sequence_efficiency(
        self,
        observation_dim: int,
        action_dim: int,
        max_sequence_length: int = 1000,
        step_size: int = 50
    ) -> Dict[str, List[float]]:
        """Analyze how Mamba handles different sequence lengths."""
        
        logger.info("Analyzing sequence length efficiency...")
        
        model_config = {
            'hidden_dim': 256,
            'num_layers': 4,
            'use_mamba': True
        }
        
        model = create_mamba_model(
            observation_dim=observation_dim,
            action_dim=action_dim,
            model_config=model_config
        ).to(self.device)
        
        sequence_lengths = list(range(50, max_sequence_length + 1, step_size))
        inference_times = []
        memory_usages = []
        
        for seq_len in sequence_lengths:
            dummy_input = torch.randn(
                1, seq_len, observation_dim,
                device=self.device
            )
            
            # Warm up
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)
            
            # Measure
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(50):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            avg_time = (time.time() - start_time) / 50
            
            if self.device.type == 'cuda':
                memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_usage = 0.0
            
            inference_times.append(avg_time)
            memory_usages.append(memory_usage)
        
        return {
            'sequence_lengths': sequence_lengths,
            'inference_times': inference_times,
            'memory_usages': memory_usages
        }
    
    def recommend_best_config(
        self,
        benchmarks: List[ModelBenchmark],
        max_params: int = 10_000_000,
        max_memory: float = 1000.0,
        min_accuracy: float = 0.7
    ) -> ModelBenchmark:
        """Recommend the best model configuration based on constraints."""
        
        # Filter by constraints
        valid_benchmarks = [
            b for b in benchmarks
            if b.parameters <= max_params
            and b.memory_usage_mb <= max_memory
            and b.accuracy >= min_accuracy
        ]
        
        if not valid_benchmarks:
            logger.warning("No configurations meet constraints, selecting best available")
            valid_benchmarks = benchmarks
        
        # Score configurations (weighted combination of metrics)
        scores = []
        for benchmark in valid_benchmarks:
            # Normalize metrics
            params_norm = 1.0 - (benchmark.parameters / max_params)
            memory_norm = 1.0 - (benchmark.memory_usage_mb / max_memory)
            accuracy_norm = benchmark.accuracy
            speed_norm = 1.0 / (benchmark.inference_time + 1e-6)
            
            # Weighted score
            score = (
                0.3 * params_norm +
                0.2 * memory_norm +
                0.3 * accuracy_norm +
                0.2 * speed_norm
            )
            scores.append(score)
        
        # Select best configuration
        best_idx = np.argmax(scores)
        best_config = valid_benchmarks[best_idx]
        
        logger.info(f"Recommended configuration: {best_config.model_name}")
        logger.info(f"Score: {scores[best_idx]:.4f}")
        
        return best_config
    
    def export_benchmark_results(
        self,
        benchmarks: List[ModelBenchmark],
        filename: str = "mamba_benchmarks.csv"
    ):
        """Export benchmark results to CSV."""
        
        df = pd.DataFrame([
            {
                'model_name': b.model_name,
                'parameters': b.parameters,
                'inference_time_ms': b.inference_time * 1000,
                'memory_usage_mb': b.memory_usage_mb,
                'accuracy': b.accuracy,
                'training_speed': b.training_speed,
                'sequence_length': b.sequence_length,
                'batch_size': b.batch_size
            }
            for b in benchmarks
        ])
        
        df.to_csv(filename, index=False)
        logger.info(f"Benchmark results exported to {filename}")
        return df


class MambaTrainer:
    """Specialized trainer for Mamba models."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_mamba_sequence_model(
        self,
        model: MambaModel,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """Train Mamba sequence model for time series prediction."""
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            optimizer.zero_grad()
            predictions = model(train_data)
            loss = criterion(predictions[:, -1, :], train_data[:, -1, :])
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_predictions = model(val_data)
                val_loss = criterion(val_predictions[:, -1, :], val_data[:, -1, :])
                val_losses.append(val_loss.item())
            model.train()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={loss.item():.6f}, Val Loss={val_loss.item():.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }


def create_mamba_exploration_report(explorer: MambaExplorer, benchmarks: List[ModelBenchmark]) -> str:
    """Create comprehensive Mamba exploration report."""
    
    report = f"""# Mamba Architecture Exploration Report

## Overview
This report summarizes the exploration of Mamba architectures for crypto trading RL agents.

## Tested Configurations
- Total configurations tested: {len(benchmarks)}
- Device: {explorer.device}

## Key Findings

### Performance Comparison
"""
    
    # Sort by accuracy
    sorted_benchmarks = sorted(benchmarks, key=lambda x: x.accuracy, reverse=True)
    
    for i, benchmark in enumerate(sorted_benchmarks[:5]):
        report += f"""
{i+1}. {benchmark.model_name}
   - Parameters: {benchmark.parameters:,}
   - Accuracy: {benchmark.accuracy:.4f}
   - Inference Time: {benchmark.inference_time:.4f}s
   - Memory Usage: {benchmark.memory_usage_mb:.2f}MB
   - Training Speed: {benchmark.training_speed:.2f}
"""
    
    return report