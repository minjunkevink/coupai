import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from typing import Tuple, List

class CoupPPO(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = self.policy(state)
        state_value = self.value(state)
        return action_probs, state_value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        action_probs, state_value = self(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), action_probs[0, action.item()], state_value

class CoupAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.policy = CoupPPO(state_dim, action_dim)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.ppo_epochs = 4
        self.clip_param = 0.2
        
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, action_prob, state_value = self.policy.get_action(state)
        return action, action_prob, state_value
    
    def update(self, states: List[np.ndarray], actions: List[int], 
               rewards: List[float], next_states: List[np.ndarray], 
               dones: List[bool], old_action_probs: List[torch.Tensor]) -> Tuple[float, float]:
        # Convert lists to numpy arrays first to avoid the warning
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        old_action_probs = torch.stack(old_action_probs)
        
        # Calculate returns and advantages
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            next_values = next_values.squeeze()
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - self.policy.value(states).squeeze()
        
        # PPO update
        policy_losses = []
        value_losses = []
        
        for _ in range(self.ppo_epochs):
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_action_probs = dist.probs[torch.arange(len(actions)), actions]
            
            # Calculate ratio
            ratio = new_action_probs / old_action_probs
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(state_values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Optimize
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        return np.mean(policy_losses), np.mean(value_losses) 