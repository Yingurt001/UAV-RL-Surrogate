"""
Step 4: Visualize Agent Behavior
================================
See your trained agent fly! This script generates:
  1. Flight trajectory plots (where does the drone go?)
  2. State evolution over time (how do position/angle change?)
  3. Action profiles (what does the policy actually output?)

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --model results/best_model.zip
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.quadrotor2d import Quadrotor2DEnv


def rollout(model, env, n_episodes=5):
    """Collect trajectories from the trained agent."""
    trajectories = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        traj = {"states": [], "actions": [], "rewards": [], "target": obs[6:8].copy()}
        done = False
        while not done:
            traj["states"].append(obs[:6].copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            traj["actions"].append(action.copy())
            traj["rewards"].append(reward)
            done = terminated or truncated
        traj["states"].append(obs[:6].copy())
        for k in ["states", "actions", "rewards"]:
            traj[k] = np.array(traj[k])
        trajectories.append(traj)
    return trajectories


def plot_trajectories(trajectories):
    """Plot 2D flight paths."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('UAV Flight Trajectories')

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    for i, traj in enumerate(trajectories):
        states = traj["states"]
        target = traj["target"]

        # Flight path
        ax.plot(states[:, 0], states[:, 1], '-', color=colors[i],
                linewidth=1.5, alpha=0.8, label=f'Episode {i+1}')
        # Start point
        ax.plot(states[0, 0], states[0, 1], 'o', color=colors[i], markersize=8)
        # End point
        ax.plot(states[-1, 0], states[-1, 1], 's', color=colors[i], markersize=8)
        # Target
        ax.plot(target[0], target[1], '*', color=colors[i], markersize=15)

    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig('results/flight_trajectories.png', dpi=150)
    plt.close(fig)
    print("Saved results/flight_trajectories.png")


def plot_state_evolution(traj):
    """Plot how each state variable evolves over one episode."""
    states = traj["states"]
    actions = traj["actions"]
    dt = 0.02
    t = np.arange(len(states)) * dt

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('State & Action Evolution (Single Episode)', fontsize=14)

    dim_names = ['x (m)', 'y (m)', 'θ (rad)', 'vx (m/s)', 'vy (m/s)', 'ω (rad/s)']

    for i, (ax, name) in enumerate(zip(axes.flat, dim_names)):
        ax.plot(t, states[:, i], 'b-', linewidth=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if i >= 4:
            ax.set_xlabel('Time (s)')

        # Mark target for x and y
        if i == 0:
            ax.axhline(y=traj["target"][0], color='r', linestyle='--',
                       label=f'Target={traj["target"][0]:.2f}')
            ax.legend()
        elif i == 1:
            ax.axhline(y=traj["target"][1], color='r', linestyle='--',
                       label=f'Target={traj["target"][1]:.2f}')
            ax.legend()

    fig.tight_layout()
    fig.savefig('results/state_evolution.png', dpi=150)
    plt.close(fig)
    print("Saved results/state_evolution.png")

    # Action plot
    if len(actions) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        t_a = np.arange(len(actions)) * dt
        ax.plot(t_a, actions[:, 0], 'b-', label='T1 (left motor)', linewidth=1.5)
        ax.plot(t_a, actions[:, 1], 'r-', label='T2 (right motor)', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Thrust')
        ax.set_title('Motor Commands')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        fig.tight_layout()
        fig.savefig('results/action_profile.png', dpi=150)
        plt.close(fig)
        print("Saved results/action_profile.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/best_model.zip")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    model = PPO.load(args.model)
    env = Quadrotor2DEnv(randomize_params=False)

    trajectories = rollout(model, env, n_episodes=8)
    plot_trajectories(trajectories)
    if len(trajectories) > 0:
        # Use the longest trajectory for detailed analysis
        longest = max(trajectories, key=lambda t: len(t["states"]))
        plot_state_evolution(longest)

    env.close()
    print("\nAll visualizations saved to results/")
