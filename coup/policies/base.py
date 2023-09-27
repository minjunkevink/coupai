from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Tuple, Dict, Any

class BasePolicy(ABC):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select an action given the current state."""
        pass
    
    @abstractmethod
    def update(self, states: list, actions: list, rewards: list, 
               next_states: list, dones: list, old_action_probs: list) -> Dict[str, float]:
        """Update the policy using the collected experiences."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the policy to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the policy from disk."""
        pass
    
    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor and move to device."""
        return torch.FloatTensor(data).to(self.device)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        return tensor.detach().cpu().numpy() 