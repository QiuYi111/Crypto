"""RL Agent implementation for crypto trading."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from loguru import logger
import pandas as pd

from .models import create_mamba_model, MambaPolicyNetwork
from .environment import CryptoTradingEnvironment
from ..config.settings import Settings


class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, transition: Tuple):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class CryptoRLAgent:
    """Main RL agent for crypto trading."""
    
    def __init__(
        self,
        settings: Settings,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_type: str = "mamba",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5
    ):
        self.settings = settings
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Determine action dimensions
        if isinstance(action_space, gym.spaces.Box):
            self.continuous_actions = True
            self.action_dim = action_space.shape[0]
        else:
            self.continuous_actions = False
            self.action_dim = action_space.n
        
        # Initialize model
        model_config = {
            'hidden_dim': settings.rl_hidden_dim,
            'num_layers': settings.rl_num_layers,
            'use_mamba': settings.rl_use_mamba,
            'continuous_actions': self.continuous_actions
        }
        
        self.policy = create_mamba_model(
            observation_space.shape[0],
            self.action_dim,
            model_config
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=settings.rl_replay_buffer_size)
        
        # Training metrics
        self.training_steps = 0
        self.episode_rewards = []
        
    def select_action(self, observation: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float]:
        """Select action using policy."""
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
            
            if self.continuous_actions:
                actions, log_probs, _ = self.policy(obs_tensor)
                action = actions.squeeze(0).cpu().numpy()
                log_prob = log_probs.squeeze(0).cpu().numpy()
            else:
                action_logits, _, _ = self.policy(obs_tensor)
                probs = torch.softmax(action_logits, dim=-1)
                
                if training:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().cpu().numpy()
                    log_prob = dist.log_prob(torch.tensor(action)).cpu().numpy()
                else:
                    action = torch.argmax(probs, dim=-1).cpu().numpy()
                    log_prob = 0.0
            
            return action, log_prob
    
    def store_transition(self, transition: Tuple):
        """Store transition in replay buffer."""
        self.replay_buffer.push(transition)
    
    def update_policy(self, batch_size: int = 64) -> Dict[str, float]:
        """Update policy using PPO."""
        
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        observations, actions, rewards, next_observations, dones, old_log_probs = zip(*batch)
        
        # Convert to tensors
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_observations = torch.FloatTensor(np.array(next_observations))
        dones = torch.FloatTensor(np.array(dones))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        
        # Ensure correct shape (batch_size, sequence_length, features)
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
            next_observations = next_observations.unsqueeze(1)
        
        # Get current predictions
        if self.continuous_actions:
            new_actions, new_log_probs, _ = self.policy(observations, actions)
            values = self.policy.get_value(observations)
        else:
            action_logits, values, new_log_probs = self.policy(observations, actions.long())
        
        # Calculate advantages
        with torch.no_grad():
            next_values = self.policy.get_value(next_observations)
            advantages = rewards + self.gamma * next_values.squeeze() * (1 - dones) - values.squeeze()
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        if self.continuous_actions:
            # Policy loss for continuous actions
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy loss
            entropy_loss = -new_log_probs.mean()
        else:
            # Policy loss for discrete actions
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            entropy_loss = -dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), rewards + self.gamma * next_values.squeeze() * (1 - dones))
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.training_steps += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
    
    def train_episode(self, env: gym.Env) -> Dict[str, float]:
        """Train for one episode."""
        
        observation, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        while True:
            # Select action
            action, log_prob = self.select_action(observation, training=True)
            
            # Execute action
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            done = terminated or truncated
            self.store_transition((
                observation,
                action,
                reward,
                next_observation,
                done,
                log_prob
            ))
            
            # Update policy
            if episode_steps % 4 == 0:  # Update every 4 steps
                metrics = self.update_policy()
            
            episode_reward += reward
            episode_steps += 1
            observation = next_observation
            
            if done:
                break
        
        self.episode_rewards.append(episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'final_balance': info.get('balance', 0.0),
            'final_portfolio_value': info.get('portfolio_value', 0.0)
        }
    
    def evaluate(self, env: gym.Env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance."""
        
        episode_rewards = []
        final_values = []
        
        for episode in range(num_episodes):
            observation, info = env.reset()
            episode_reward = 0.0
            
            while True:
                action, _ = self.select_action(observation, training=False)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            final_values.append(info.get('portfolio_value', 0.0))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'win_rate': np.mean([r > 0 for r in episode_rewards])
        }
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'settings': self.settings
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.episode_rewards = checkpoint['episode_rewards']
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        
        return {
            'total_episodes': len(self.episode_rewards),
            'training_steps': self.training_steps,
            'mean_reward': np.mean(self.episode_rewards),
            'recent_mean_reward': np.mean(recent_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards)
        }