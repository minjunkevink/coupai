from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

class Character(Enum):
    DUKE = 0
    ASSASSIN = 1
    CAPTAIN = 2
    AMBASSADOR = 3
    CONTESSA = 4

class Action(Enum):
    INCOME = 0
    FOREIGN_AID = 1
    COUP = 2
    TAXES = 3
    ASSASSINATE = 4
    STEAL = 5
    EXCHANGE = 6
    BLOCK = 7
    CHALLENGE = 8

@dataclass
class Player:
    id: int
    coins: int
    influence: List[Character]
    is_alive: bool = True

class CoupGame:
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.players: List[Player] = []
        self.deck: List[Character] = []
        self.bank: int = 50  # Starting bank coins
        self.current_player: int = 0
        self.last_action: Optional[Action] = None
        self.last_target: Optional[int] = None
        self.setup_game()

    def setup_game(self):
        # Initialize deck with 3 copies of each character
        self.deck = [char for char in Character] * 3
        np.random.shuffle(self.deck)

        # Initialize players
        for i in range(self.num_players):
            influence = [self.deck.pop() for _ in range(2)]
            self.players.append(Player(id=i, coins=2, influence=influence))

    def get_valid_actions(self, player_id: int) -> List[Action]:
        player = self.players[player_id]
        if not player.is_alive:
            return []

        actions = [Action.INCOME, Action.FOREIGN_AID]
        
        if player.coins >= 7:
            actions.append(Action.COUP)
        if player.coins >= 3:
            actions.append(Action.ASSASSINATE)
        
        # Add character-specific actions
        for char in player.influence:
            if char == Character.DUKE:
                actions.append(Action.TAXES)
            elif char == Character.CAPTAIN:
                actions.append(Action.STEAL)
            elif char == Character.AMBASSADOR:
                actions.append(Action.EXCHANGE)

        return actions

    def get_state(self) -> np.ndarray:
        """Convert game state to numpy array for RL"""
        state = []
        for player in self.players:
            state.extend([
                player.coins,
                len(player.influence),
                player.is_alive
            ])
        state.append(self.bank)
        return np.array(state)

    def step(self, action: Action, target: Optional[int] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the game
        Returns: (next_state, reward, done, info)
        """
        player = self.players[self.current_player]
        reward = 0
        done = False
        info = {}

        if action == Action.INCOME:
            player.coins += 1
            self.bank -= 1
        elif action == Action.FOREIGN_AID:
            player.coins += 2
            self.bank -= 2
        elif action == Action.COUP:
            if target is None:
                raise ValueError("Target required for Coup")
            target_player = self.players[target]
            if len(target_player.influence) > 0:
                target_player.influence.pop()
                if len(target_player.influence) == 0:
                    target_player.is_alive = False
            player.coins -= 7
            self.bank += 7

        # Check if game is over
        alive_players = [p for p in self.players if p.is_alive]
        if len(alive_players) == 1:
            done = True
            reward = 1.0 if alive_players[0].id == self.current_player else -1.0

        # Move to next player
        self.current_player = (self.current_player + 1) % self.num_players
        while not self.players[self.current_player].is_alive:
            self.current_player = (self.current_player + 1) % self.num_players

        return self.get_state(), reward, done, info

    def reset(self) -> np.ndarray:
        """Reset the game to initial state"""
        self.__init__(self.num_players)
        return self.get_state() 