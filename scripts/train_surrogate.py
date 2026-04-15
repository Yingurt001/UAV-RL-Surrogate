"""
Step 2: Collect Data & Train Surrogate Dynamics Model
=====================================================
This script does two things:
  1. Collects (state, action, next_state) transition data from the real env
  2. Trains a neural network to predict these transitions

This is the CORE of surrogate modeling: replace expensive/unknown physics
with a learned approximation.

=== THE DATA COLLECTION STRATEGY ===

We collect data from TWO sources:
  A) The trained PPO agent — shows "expert" behavior near good states
  B) Random actions — covers the full state space, including states
     the expert avoids

=== THINK ABOUT THIS ===
Q1: Why not just use expert data?
    -> Expert data is biased toward "good" states. If the surrogate only
       sees those states, it can't predict what happens in "bad" states
       (e.g., what if the drone tilts 45°?). During RL training on the
       surrogate, the agent WILL visit bad states early on, and the
       surrogate must handle them.

Q2: What's the train/val split for?
    -> To detect overfitting. If train_loss << val_loss, the model
       memorized the data rather than learning the dynamics.

Usage:
    python scripts/train_surrogate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.quadrotor2d import Quadrotor2DEnv
from models.surrogate import SurrogateModel


def collect_data(n_episodes=200, mix_ratio=0.5):
    """
    Collect transition data: (state, action, next_state).

    mix_ratio: fraction of episodes using the trained PPO agent.
    The rest use random actions for coverage.

    === THINK ABOUT THIS ===
    Q3: We use randomize_params=True during collection. Why?
        -> So the surrogate sees transitions under DIFFERENT physical
           parameters. It must learn a general dynamics model, not one
           that only works for mass=0.5kg.
    """
    env = Quadrotor2DEnv(randomize_params=True)

    # Load trained PPO agent (if available)
    ppo_model = None
    if os.path.exists("results/best_model.zip"):
        ppo_model = PPO.load("results/best_model")
        print("Loaded trained PPO agent for data collection")

    states, actions, next_states = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        # Use PPO for mix_ratio fraction of episodes, random for the rest
        use_expert = ppo_model is not None and ep < int(n_episodes * mix_ratio)
        done = False

        while not done:
            state_6d = obs[:6]  # extract state (without target)

            if use_expert:
                action, _ = ppo_model.predict(obs, deterministic=False)
            else:
                action = env.action_space.sample()

            next_obs, _, terminated, truncated, _ = env.step(action)
            next_state_6d = next_obs[:6]

            states.append(state_6d)
            actions.append(action)
            next_states.append(next_state_6d)

            obs = next_obs
            done = terminated or truncated

    env.close()

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)

    print(f"Collected {len(states):,} transitions from {n_episodes} episodes")
    print(f"  Expert episodes: {int(n_episodes * mix_ratio)}")
    print(f"  Random episodes: {n_episodes - int(n_episodes * mix_ratio)}")
    return states, actions, next_states


def train_surrogate(states, actions, next_states, epochs=100, batch_size=256, lr=1e-3):
    """
    Train the surrogate model on collected data.

    === THINK ABOUT THIS ===
    Q4: We predict DELTA (next_state - state), not next_state. Why again?
        -> Example: state=[2.0, 3.0, ...], next_state=[2.001, 3.002, ...]
           Predicting [2.001, 3.002] requires the network to output large
           numbers precisely. Predicting [0.001, 0.002] is much easier.
           This is residual learning — one of the most important tricks
           in neural network dynamics modeling.
    """
    # Compute deltas (residual targets)
    deltas = next_states - states

    # Train/val split (80/20)
    n = len(states)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    # Create model
    model = SurrogateModel(state_dim=6, action_dim=2)
    model.set_normalization(states[train_idx], actions[train_idx], deltas[train_idx])

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(states[train_idx]),
        torch.tensor(actions[train_idx]),
        torch.tensor(deltas[train_idx]),
    )
    val_ds = TensorDataset(
        torch.tensor(states[val_idx]),
        torch.tensor(actions[val_idx]),
        torch.tensor(deltas[val_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"\nTraining surrogate model...")
    print(f"  Train samples: {split:,}, Val samples: {n - split:,}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0
        for s, a, d in train_loader:
            pred_delta = model(s, a)
            loss = criterion(pred_delta, d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(s)
        train_loss = epoch_loss / split
        train_losses.append(train_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for s, a, d in val_loader:
                pred_delta = model(s, a)
                val_loss += criterion(pred_delta, d).item() * len(s)
        val_loss /= (n - split)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/surrogate_best.pth")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                  f"{'  *best*' if val_loss == best_val_loss else ''}")

    # Load best model
    model.load_state_dict(torch.load("results/surrogate_best.pth", weights_only=True))

    # Save final model with normalization stats
    torch.save(model.state_dict(), "results/surrogate_final.pth")
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to results/surrogate_final.pth")

    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    analyze_predictions(model, states[val_idx], actions[val_idx], deltas[val_idx])

    return model


def plot_training_curves(train_losses, val_losses):
    """Plot and save training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Surrogate Model Training')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/surrogate_training_curve.png', dpi=150)
    plt.close(fig)
    print("Training curve saved to results/surrogate_training_curve.png")


def analyze_predictions(model, states, actions, deltas):
    """
    Analyze prediction quality per state dimension.

    === THINK ABOUT THIS ===
    Q5: Why analyze per dimension?
        -> Some dimensions are easier to predict than others. Position
           changes are smooth and predictable. Angular velocity can be
           chaotic. Knowing WHERE the model struggles tells you what
           to improve (more data? bigger network? better features?).
    """
    model.eval()
    with torch.no_grad():
        s = torch.tensor(states)
        a = torch.tensor(actions)
        pred = model(s, a).numpy()

    dim_names = ['dx', 'dy', 'dtheta', 'dvx', 'dvy', 'domega']
    true = deltas

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Surrogate Prediction Quality (per dimension)', fontsize=14)

    for i, (ax, name) in enumerate(zip(axes.flat, dim_names)):
        ax.scatter(true[:, i], pred[:, i], alpha=0.1, s=1)
        lims = [min(true[:, i].min(), pred[:, i].min()),
                max(true[:, i].max(), pred[:, i].max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='perfect')
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # R² score
        ss_res = np.sum((true[:, i] - pred[:, i]) ** 2)
        ss_tot = np.sum((true[:, i] - true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        ax.text(0.05, 0.95, f'R²={r2:.4f}', transform=ax.transAxes,
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    fig.tight_layout()
    fig.savefig('results/surrogate_prediction_quality.png', dpi=150)
    plt.close(fig)
    print("Prediction quality plot saved to results/surrogate_prediction_quality.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Step 1: Collect data
    states, actions, next_states = collect_data(n_episodes=300, mix_ratio=0.5)

    # Step 2: Train surrogate
    model = train_surrogate(states, actions, next_states, epochs=100)
