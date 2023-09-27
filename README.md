# Coup RL Agent

This project implements a Reinforcement Learning agent for the card game Coup using Proximal Policy Optimization (PPO).

## Project Structure

- `coup/`
  - `game.py`: Core game implementation
  - `env.py`: Gymnasium environment wrapper
  - `agent.py`: PPO agent implementation
- `train.py`: Training script
- `requirements.txt`: Project dependencies

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the agent:

```bash
python train.py
```

The training script will:
- Create a new environment with 3 players
- Train the agent for 10,000 episodes
- Save model checkpoints every 1000 episodes
- Log training metrics to TensorBoard

You can monitor training progress using TensorBoard:
```bash
tensorboard --logdir runs/coup_ppo
```

## Game Rules

The game follows the standard Coup rules:
- 3-6 players
- Each player starts with 2 influence cards and 2 coins
- Players take turns performing actions
- Actions can be challenged or blocked
- Last player with influence cards wins

## State and Action Space

The agent observes:
- Each player's coins
- Number of influence cards
- Alive/dead status
- Bank coins

Actions include:
- Income
- Foreign Aid
- Coup
- Taxes (Duke)
- Assassinate (Assassin)
- Steal (Captain)
- Exchange (Ambassador)
- Block
- Challenge

## Model Architecture

The agent uses a PPO implementation with:
- Policy network: 3-layer neural network
- Value network: 3-layer neural network
- Hidden layer size: 128
- Learning rate: 3e-4
- PPO epochs: 4
- Clip parameter: 0.2 