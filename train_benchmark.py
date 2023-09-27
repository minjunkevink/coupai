import argparse
import os
import torch
import numpy as np
import wandb
from tqdm import tqdm
from coup.env import CoupEnv
from coup.policies import get_policy
from coup.config import load_config

def evaluate(env, policy, num_episodes: int) -> dict:
    """Evaluate the policy over multiple episodes."""
    rewards = []
    lengths = []
    win_rates = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = policy.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        win_rates.append(1.0 if info.get('winner') == 0 else 0.0)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'win_rate': np.mean(win_rates)
    }

def train(config_path: str):
    """Train the policy with the given configuration."""
    # Load configuration
    config = load_config(config_path)
    
    # Set random seeds
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    # Create environment and policy
    env = CoupEnv(config.game.num_players)
    policy = get_policy(config.policy.name, env)
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.run_name,
        config=vars(config),
        tags=config.tags
    )
    
    # Training loop
    best_reward = float('-inf')
    for episode in tqdm(range(config.training.num_episodes)):
        # Collect episode
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        # Update policy
        metrics = policy.update()
        
        # Log metrics
        wandb.log({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'win_rate': 1.0 if info.get('winner') == 0 else 0.0,
            **metrics
        })
        
        # Evaluate periodically
        if episode % config.training.eval_freq == 0:
            eval_metrics = evaluate(env, policy, config.training.num_eval_episodes)
            wandb.log({
                'eval_mean_reward': eval_metrics['mean_reward'],
                'eval_std_reward': eval_metrics['std_reward'],
                'eval_mean_length': eval_metrics['mean_length'],
                'eval_win_rate': eval_metrics['win_rate']
            })
            
            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                policy.save(f"models/{config.run_name}_best.pt")
        
        # Save checkpoint periodically
        if episode % config.training.save_freq == 0:
            policy.save(f"models/{config.run_name}_checkpoint_{episode}.pt")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    train(args.config) 