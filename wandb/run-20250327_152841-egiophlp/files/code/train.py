import numpy as np
from coup.env import CoupEnv
from coup.agent import CoupAgent
from tqdm import tqdm
import torch
import wandb
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

def create_action_distribution_plot(action_counts, num_players):
    plt.figure(figsize=(12, 6))
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    
    # Create grouped bar chart
    x = np.arange(len(actions))
    width = 0.8 / num_players
    
    for i in range(num_players):
        player_counts = [action_counts[action][i] for action in actions]
        plt.bar(x + i*width, player_counts, width, label=f'Player {i}')
    
    plt.xlabel('Actions')
    plt.ylabel('Count')
    plt.title('Action Distribution by Player')
    plt.xticks(x + width*(num_players-1)/2, actions, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Convert plot to image
    plt.savefig('action_distribution.png')
    plt.close()
    
    return wandb.Image('action_distribution.png')

def create_state_heatmap(states, num_players):
    plt.figure(figsize=(10, 6))
    # Reshape states to 2D array for heatmap
    state_matrix = np.array(states)
    sns.heatmap(state_matrix, cmap='viridis')
    plt.title('Game State Heatmap')
    plt.xlabel('State Features')
    plt.ylabel('Time Steps')
    plt.tight_layout()
    
    plt.savefig('state_heatmap.png')
    plt.close()
    
    return wandb.Image('state_heatmap.png')

def create_reward_trajectory_plot(rewards_history):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title('Reward Trajectory')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('reward_trajectory.png')
    plt.close()
    
    return wandb.Image('reward_trajectory.png')

def collect_episode(env, agent, num_players):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    # Store episode data
    states, actions, rewards, next_states, dones, action_probs = [], [], [], [], [], []
    action_counts = defaultdict(lambda: [0] * num_players)
    player_coins = defaultdict(list)
    player_influence = defaultdict(list)
    
    while not done:
        # Select action
        action, action_prob, _ = agent.select_action(state)
        action_type = action // num_players
        target = action % num_players
        
        # Record action
        action_name = env.game.Action(action_type).name
        action_counts[action_name][env.game.current_player] += 1
        
        # Record player states
        for i in range(num_players):
            player = env.game.players[i]
            player_coins[f'player_{i}_coins'].append(player.coins)
            player_influence[f'player_{i}_influence'].append(len(player.influence))
        
        # Take action
        next_state, reward, done, info = env.step([action_type, target])
        
        # Store transition
        states.append(state)
        actions.append(action)
        rewards.append(next_state)
        next_states.append(next_state)
        dones.append(done)
        action_probs.append(action_prob)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        if done:
            break
    
    return (states, actions, rewards, next_states, dones, action_probs, 
            episode_reward, episode_length, action_counts, player_coins, player_influence)

def train(num_episodes: int = 10000, num_players: int = 3, episodes_per_update: int = 4):
    # Initialize wandb
    wandb.init(
        project="coup-ai",
        config={
            "num_episodes": num_episodes,
            "num_players": num_players,
            "episodes_per_update": episodes_per_update,
            "learning_rate": 3e-4,
            "ppo_epochs": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "hidden_dim": 128
        }
    )
    
    # Create environment and agent
    env = CoupEnv(num_players)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.nvec[0] * env.action_space.nvec[1]
    agent = CoupAgent(state_dim, action_dim)
    
    # Initialize tracking variables
    rewards_history = []
    win_rates = defaultdict(list)
    action_distribution = defaultdict(lambda: [0] * num_players)
    
    # Training loop
    for episode in tqdm(range(num_episodes)):
        # Collect multiple episodes before updating
        all_states, all_actions, all_rewards, all_next_states, all_dones, all_action_probs = [], [], [], [], [], []
        total_reward = 0
        total_length = 0
        episode_action_counts = defaultdict(lambda: [0] * num_players)
        episode_player_coins = defaultdict(list)
        episode_player_influence = defaultdict(list)
        
        for _ in range(episodes_per_update):
            (states, actions, rewards, next_states, dones, action_probs, 
             episode_reward, episode_length, action_counts, player_coins, player_influence) = collect_episode(env, agent, num_players)
            
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_next_states.extend(next_states)
            all_dones.extend(dones)
            all_action_probs.extend(action_probs)
            total_reward += episode_reward
            total_length += episode_length
            
            # Update action counts
            for action, counts in action_counts.items():
                for i in range(num_players):
                    episode_action_counts[action][i] += counts[i]
            
            # Update player states
            for key, values in player_coins.items():
                episode_player_coins[key].extend(values)
            for key, values in player_influence.items():
                episode_player_influence[key].extend(values)
        
        # Update agent with collected experiences
        policy_loss, value_loss = agent.update(all_states, all_actions, all_rewards, all_next_states, all_dones, all_action_probs)
        
        # Calculate metrics
        avg_reward = total_reward / episodes_per_update
        avg_length = total_length / episodes_per_update
        rewards_history.append(avg_reward)
        
        # Calculate win rates
        for i in range(num_players):
            wins = sum(1 for p in env.game.players if p.is_alive and p.id == i)
            win_rate = wins / episodes_per_update
            win_rates[f'player_{i}_win_rate'].append(win_rate)
        
        # Create visualizations
        action_dist_plot = create_action_distribution_plot(episode_action_counts, num_players)
        state_heatmap_plot = create_state_heatmap(all_states, num_players)
        reward_trajectory_plot = create_reward_trajectory_plot(rewards_history)
        
        # Log metrics and visualizations to wandb
        log_dict = {
            "episode": episode,
            "average_reward": avg_reward,
            "average_episode_length": avg_length,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_reward": total_reward,
            "total_episode_length": total_length,
            "action_distribution": action_dist_plot,
            "state_heatmap": state_heatmap_plot,
            "reward_trajectory": reward_trajectory_plot
        }
        
        # Add player-specific metrics
        for i in range(num_players):
            log_dict.update({
                f"player_{i}_win_rate": win_rates[f'player_{i}_win_rate'][-1],
                f"player_{i}_avg_coins": np.mean(episode_player_coins[f'player_{i}_coins']),
                f"player_{i}_avg_influence": np.mean(episode_player_influence[f'player_{i}_influence']),
                f"player_{i}_max_coins": np.max(episode_player_coins[f'player_{i}_coins']),
                f"player_{i}_min_coins": np.min(episode_player_coins[f'player_{i}_coins'])
            })
        
        wandb.log(log_dict)
        
        # Save model periodically
        if episode % 100 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'models/coup_ppo_{episode}_{timestamp}.pt'
            torch.save(agent.policy.state_dict(), model_path)
            wandb.save(model_path)
    
    wandb.finish()

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train()