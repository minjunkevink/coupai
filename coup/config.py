from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
import os

@dataclass
class GameConfig:
    num_players: int  # 3-6 players
    starting_coins: int = 2  # Each player starts with 2 coins
    bank_coins: int = 50  # Starting bank coins
    influence_cards_per_player: int = 2  # Each player starts with 2 influence cards
    copies_per_character: int = 3  # 3 copies of each character in the deck
    coup_cost: int = 7  # Cost to perform a coup
    assassinate_cost: int = 3  # Cost to perform an assassination
    max_coins: int = 10  # Maximum coins a player can have
    forced_coup_threshold: int = 10  # Must coup if you have 10 or more coins

@dataclass
class PolicyConfig:
    name: str
    hidden_dims: list
    activation: str
    learning_rate: float
    gamma: float
    batch_size: int
    # Policy-specific parameters
    ppo_epochs: Optional[int] = None
    clip_param: Optional[float] = None
    value_loss_coef: Optional[float] = None
    dqn_buffer_size: Optional[int] = None
    target_update_freq: Optional[int] = None
    exploration_rate: Optional[float] = None
    sac_temperature: Optional[float] = None
    target_entropy: Optional[float] = None

@dataclass
class TrainingConfig:
    num_episodes: int
    eval_freq: int
    save_freq: int
    num_eval_episodes: int
    seed: int
    device: str
    # Training-specific parameters
    reward_scale: float = 1.0  # Scale rewards for better learning
    max_episode_length: int = 100  # Maximum steps per episode
    num_parallel_envs: int = 1  # Number of parallel environments for training

@dataclass
class ExperimentConfig:
    game: GameConfig
    policy: PolicyConfig
    training: TrainingConfig
    wandb_project: str
    wandb_entity: str
    run_name: str

def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ExperimentConfig(
        game=GameConfig(**config_dict['game']),
        policy=PolicyConfig(**config_dict['policy']),
        training=TrainingConfig(**config_dict['training']),
        wandb_project=config_dict['wandb_project'],
        wandb_entity=config_dict['wandb_entity'],
        run_name=config_dict['run_name']
    )

def save_config(config: ExperimentConfig, save_dir: str):
    """Save configuration to YAML file."""
    os.makedirs(save_dir, exist_ok=True)
    config_dict = {
        'game': vars(config.game),
        'policy': vars(config.policy),
        'training': vars(config.training),
        'wandb_project': config.wandb_project,
        'wandb_entity': config.wandb_entity,
        'run_name': config.run_name
    }
    
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False) 