import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .game import CoupGame, Action

class CoupEnv(gym.Env):
    def __init__(self, num_players: int = 3):
        super().__init__()
        self.num_players = num_players
        self.game = CoupGame(num_players)
        
        # Define action space
        # For each action, we need to specify:
        # 1. The action type (9 possible actions)
        # 2. The target player (num_players possible targets)
        self.action_space = spaces.MultiDiscrete([
            len(Action),  # Action type
            num_players   # Target player
        ])
        
        # Define observation space
        # For each player we observe:
        # - coins (0-10)
        # - number of influence cards (0-2)
        # - alive status (0-1)
        # Plus bank coins (0-50)
        obs_size = num_players * 3 + 1
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(obs_size,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game = CoupGame(self.num_players)
        return self.game.get_state(), {}

    def step(self, action):
        action_type, target = action
        
        # Convert action index to Action enum
        action_enum = Action(action_type)
        
        # Get valid actions for current player
        valid_actions = self.game.get_valid_actions(self.game.current_player)
        
        # Check if action is valid
        if action_enum not in valid_actions:
            return self.game.get_state(), -1.0, True, {"invalid_action": True}
        
        # Execute action
        next_state, reward, done, info = self.game.step(action_enum, target)
        
        return next_state, reward, done, info

    def render(self):
        # Simple text-based rendering
        print("\nGame State:")
        print(f"Bank: {self.game.bank}")
        for player in self.game.players:
            print(f"Player {player.id}: {player.coins} coins, {len(player.influence)} influence cards, {'Alive' if player.is_alive else 'Dead'}")
        print(f"Current Player: {self.game.current_player}") 