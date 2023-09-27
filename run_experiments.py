import os
import subprocess
from datetime import datetime
import yaml
from itertools import product
import time
import wandb

def generate_experiment_name(policy: str, players: int, arch: dict, lr: float, batch_size: int) -> str:
    """Generate a descriptive experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_desc = f"{len(arch['hidden_dims'])}x{arch['hidden_dims'][0]}_{arch['activation']}"
    lr_desc = f"lr{lr:.0e}"
    return f"{policy}_{players}p_{arch_desc}_{lr_desc}_batch{batch_size}_{timestamp}"

def generate_configs():
    """Generate different configuration combinations."""
    configs = []
    
    # Policy types and their specific parameters
    policies = {
        "ppo": {
            "ppo_epochs": 4,
            "clip_param": 0.2,
            "value_loss_coef": 0.5
        },
        "dqn": {
            "dqn_buffer_size": 100000,
            "target_update_freq": 1000,
            "exploration_rate": 0.1
        },
        "sac": {
            "sac_temperature": 0.2,
            "target_entropy": -1.0
        }
    }
    
    # Network architectures
    architectures = [
        {"hidden_dims": [64, 64], "activation": "relu"},
        {"hidden_dims": [128, 128], "activation": "relu"},
        {"hidden_dims": [256, 256], "activation": "relu"},
        {"hidden_dims": [128, 128, 128], "activation": "relu"},
        {"hidden_dims": [128, 128], "activation": "elu"}
    ]
    
    # Learning rates
    learning_rates = [1e-4, 3e-4, 1e-5]
    
    # Batch sizes
    batch_sizes = [32, 64, 128]
    
    # Number of players (3-6)
    num_players = [3, 4, 5, 6]
    
    # Generate combinations
    for policy_name, policy_params in policies.items():
        for arch, lr, batch_size, players in product(architectures, learning_rates, batch_sizes, num_players):
            run_name = generate_experiment_name(policy_name, players, arch, lr, batch_size)
            
            config = {
                "game": {
                    "num_players": players,
                    "starting_coins": 2,
                    "bank_coins": 50,
                    "influence_cards_per_player": 2,
                    "copies_per_character": 3,
                    "coup_cost": 7,
                    "assassinate_cost": 3,
                    "max_coins": 10,
                    "forced_coup_threshold": 10
                },
                "policy": {
                    "name": policy_name,
                    "hidden_dims": arch["hidden_dims"],
                    "activation": arch["activation"],
                    "learning_rate": lr,
                    "gamma": 0.99,
                    "batch_size": batch_size,
                    **policy_params
                },
                "training": {
                    "num_episodes": 10000,
                    "eval_freq": 100,
                    "save_freq": 1000,
                    "num_eval_episodes": 10,
                    "seed": 42,
                    "device": "cuda",
                    "reward_scale": 1.0,
                    "max_episode_length": 100,
                    "num_parallel_envs": 1
                },
                "wandb_project": "coup-ai",
                "wandb_entity": "your-username",
                "run_name": run_name,
                "tags": [
                    f"policy_{policy_name}",
                    f"players_{players}",
                    f"arch_{len(arch['hidden_dims'])}x{arch['hidden_dims'][0]}",
                    f"activation_{arch['activation']}",
                    f"lr_{lr:.0e}",
                    f"batch_{batch_size}"
                ]
            }
            
            configs.append(config)
    
    return configs

def run_experiments():
    """Run all experiments with different configurations."""
    # Create configs directory
    os.makedirs("configs", exist_ok=True)
    
    # Generate configurations
    configs = generate_configs()
    
    # Save and run each configuration
    for config in configs:
        # Save configuration
        config_path = f"configs/{config['run_name']}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Run training
        print(f"Running experiment: {config['run_name']}")
        subprocess.run(["python", "train_benchmark.py", "--config", config_path])
        
        # Wait a bit between runs
        time.sleep(5)

def main():
    # Initialize wandb
    wandb.login()
    
    # Run experiments
    run_experiments()
    
    # Run analysis
    print("\nRunning analysis...")
    subprocess.run(["python", "analyze_results.py"])

if __name__ == "__main__":
    main() 