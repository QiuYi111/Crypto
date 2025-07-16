"""Mamba model architectures for RL agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger


try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba-ssm not available. Using fallback transformer architecture.")


class MambaBlock(nn.Module):
    """Mamba block for sequence modeling."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # Fallback to transformer encoder
            self.mamba = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block."""
        if MAMBA_AVAILABLE:
            residual = x
            x = self.mamba(x)
            x = self.dropout(x)
            return self.norm(x + residual)
        else:
            # Transformer fallback
            return self.mamba(x)


class MambaModel(nn.Module):
    """Mamba-based sequence model for crypto trading."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_mamba: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_mamba = use_mamba and MAMBA_AVAILABLE
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Pass through Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Final projection
        x = self.output_proj(x)
        x = self.dropout(x)
        
        return x


class MambaPolicyNetwork(nn.Module):
    """Policy network using Mamba architecture for crypto trading."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_mamba: bool = True,
        continuous_actions: bool = False
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        
        # Shared Mamba backbone
        self.backbone = MambaModel(
            input_dim=observation_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_mamba=use_mamba
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        if continuous_actions:
            # For continuous actions, output mean and log_std
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self, 
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through policy network.
        
        Args:
            observations: Tensor of shape (batch_size, seq_len, observation_dim)
            actions: Optional actions for evaluation
            
        Returns:
            Tuple of (action_logits/log_probs, values, action_log_probs)
        """
        # Get sequence representations
        hidden = self.backbone(observations)
        
        # Use last timestep for decision making
        last_hidden = hidden[:, -1, :]  # (batch_size, hidden_dim)
        
        # Actor output
        action_logits = self.actor(last_hidden)
        
        if self.continuous_actions:
            # Continuous actions: output mean and std
            mean = torch.tanh(action_logits)
            std = torch.exp(self.log_std)
            
            if actions is None:
                # Sample actions
                actions = mean + std * torch.randn_like(mean)
                action_log_probs = -0.5 * (((actions - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))
                action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
            else:
                # Evaluate given actions
                action_log_probs = -0.5 * (((actions - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))
                action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
            
            return actions, action_log_probs, None
        else:
            # Discrete actions
            action_probs = F.softmax(action_logits, dim=-1)
            
            if actions is None:
                # Sample actions
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()
                action_log_probs = dist.log_prob(actions)
            else:
                # Evaluate given actions
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(actions.squeeze(-1))
            
            return action_logits, action_log_probs, None
        
        # Critic output
        values = self.critic(last_hidden)
        
        return action_logits, values, action_log_probs
    
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Get value estimates for observations."""
        hidden = self.backbone(observations)
        last_hidden = hidden[:, -1, :]
        return self.critic(last_hidden)


class AttentionMambaHybrid(nn.Module):
    """Hybrid model combining Mamba with attention mechanism."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_mamba_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=hidden_dim, dropout=dropout)
            for _ in range(num_mamba_layers)
        ])
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid model."""
        x = self.input_proj(x)
        
        # Mamba processing
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        
        # Attention processing
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        
        return x


class MarketStateEncoder(nn.Module):
    """Encoder for market state information."""
    
    def __init__(
        self,
        market_features: int,
        confidence_features: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Market features encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(market_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Confidence features encoder
        self.confidence_encoder = nn.Sequential(
            nn.Linear(confidence_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined encoder
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self, 
        market_features: torch.Tensor, 
        confidence_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode market state information."""
        
        market_encoded = self.market_encoder(market_features)
        confidence_encoded = self.confidence_encoder(confidence_features)
        
        # Concatenate and encode
        combined = torch.cat([market_encoded, confidence_encoded], dim=-1)
        return self.combined_encoder(combined)


def create_mamba_model(
    observation_space: int,
    action_space: int,
    model_config: Dict[str, Any]
) -> MambaPolicyNetwork:
    """Factory function to create Mamba model."""
    
    return MambaPolicyNetwork(
        observation_dim=observation_space,
        action_dim=action_space,
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 4),
        use_mamba=model_config.get('use_mamba', True),
        continuous_actions=model_config.get('continuous_actions', False)
    )


class ModelRegistry:
    """Registry for different model architectures."""
    
    MODELS = {
        'mamba': MambaPolicyNetwork,
        'hybrid': AttentionMambaHybrid,
        'transformer': None,  # Will use fallback
    }
    
    @classmethod
    def get_model(cls, model_type: str, **kwargs):
        """Get model by type."""
        if model_type not in cls.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls.MODELS[model_type]
        if model_class is None:
            # Use fallback
            return MambaPolicyNetwork(**kwargs)
        
        return model_class(**kwargs)