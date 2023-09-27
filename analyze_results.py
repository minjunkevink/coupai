import wandb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any
import os

def get_experiment_data(project_name: str, entity: str) -> pd.DataFrame:
    """Fetch all experiment data from wandb."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}")
    
    data = []
    for run in runs:
        # Extract configuration from run
        config = run.config
        
        # Get metrics history
        history = run.scan_history()
        for row in history:
            data.append({
                'run_name': run.name,
                'policy': config['policy']['name'],
                'num_players': config['game']['num_players'],
                'architecture': f"{len(config['policy']['hidden_dims'])}layers_{config['policy']['activation']}",
                'learning_rate': config['policy']['learning_rate'],
                'batch_size': config['policy']['batch_size'],
                'episode': row.get('episode', 0),
                'average_reward': row.get('average_reward', 0),
                'win_rate': row.get('win_rate', 0),
                'policy_loss': row.get('policy_loss', 0),
                'value_loss': row.get('value_loss', 0),
                'entropy': row.get('entropy', 0)
            })
    
    return pd.DataFrame(data)

def create_learning_curves(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Create learning curves for different policies."""
    fig = go.Figure()
    
    for policy in df['policy'].unique():
        policy_data = df[df['policy'] == policy]
        
        # Calculate mean and std for each episode
        means = policy_data.groupby('episode')[metric].mean()
        stds = policy_data.groupby('episode')[metric].std()
        
        fig.add_trace(go.Scatter(
            x=means.index,
            y=means,
            name=policy,
            error_y=dict(type='data', array=stds, visible=True)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Episode',
        yaxis_title=metric,
        template='plotly_white'
    )
    
    return fig

def create_comparison_plots(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create all comparison plots."""
    plots = {}
    
    # Learning curves for different metrics
    metrics = {
        'average_reward': 'Average Episode Reward',
        'win_rate': 'Win Rate',
        'policy_loss': 'Policy Loss',
        'value_loss': 'Value Loss',
        'entropy': 'Policy Entropy'
    }
    
    for metric, title in metrics.items():
        plots[metric] = create_learning_curves(df, metric, title)
    
    # Final performance comparison
    final_performance = df.groupby(['policy', 'num_players'])['average_reward'].mean().reset_index()
    
    fig = go.Figure()
    for policy in final_performance['policy'].unique():
        policy_data = final_performance[final_performance['policy'] == policy]
        fig.add_trace(go.Bar(
            x=policy_data['num_players'],
            y=policy_data['average_reward'],
            name=policy
        ))
    
    fig.update_layout(
        title='Final Performance by Number of Players',
        xaxis_title='Number of Players',
        yaxis_title='Average Reward',
        template='plotly_white'
    )
    plots['final_performance'] = fig
    
    return plots

def save_plots_to_wandb(plots: Dict[str, go.Figure], project_name: str, entity: str):
    """Save plots to wandb."""
    with wandb.init(project=project_name, entity=entity, name="analysis") as run:
        for name, fig in plots.items():
            wandb.log({name: wandb.Image(fig)})

def main():
    # Configuration
    project_name = "coup-ai"
    entity = "your-username"
    
    # Get data
    df = get_experiment_data(project_name, entity)
    
    # Create plots
    plots = create_comparison_plots(df)
    
    # Save to wandb
    save_plots_to_wandb(plots, project_name, entity)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.groupby(['policy', 'num_players'])['average_reward'].agg(['mean', 'std']))

if __name__ == "__main__":
    main() 