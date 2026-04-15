"""
Step 3: The Big Comparison — Real vs. Surrogate
================================================
This is the culmination of the project. We answer the question:

  "Can an RL agent trained in a LEARNED world perform well in the REAL world?"

We train PPO in three settings:
  A) Real env    → test in Real env     (baseline, the gold standard)
  B) Surrogate   → test in Surrogate    (how good is surrogate-only training?)
  C) Surrogate   → test in Real env     (THE KEY TEST: does it transfer?)

=== THINK ABOUT THIS ===
Q1: What result would be ideal?
    -> B ≈ A: surrogate training matches real training
    -> C ≈ A: surrogate-trained policy transfers to reality
    This would mean we DON'T NEED a physics simulator at all —
    just collect some data, train a surrogate, and do RL on it.

Q2: What result would be concerning?
    -> B >> A: agent exploits surrogate inaccuracies (model exploitation)
    -> C << A: surrogate policy doesn't transfer (sim-to-real gap)

Q3: How does this relate to your UAV research?
    -> This is EXACTLY the pipeline: collect flight data from real UAV,
       train a surrogate (PyTorch NN), then train RL controllers on it.
       No physics equations needed. No parameter identification needed.

Usage:
    python scripts/compare.py
    python scripts/compare.py --timesteps 100000  # shorter training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from envs.quadrotor2d import Quadrotor2DEnv
from envs.surrogate_env import SurrogateEnv


def evaluate_agent(model, env, n_episodes=50):
    """Run agent in environment and collect per-episode rewards."""
    rewards = []
    lengths = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
    return np.array(rewards), np.array(lengths)


def train_and_evaluate(args):
    """Train in different settings and compare results."""
    os.makedirs("results", exist_ok=True)

    results = {}

    # ============================================================
    # Setting A: Train on REAL env (baseline)
    # ============================================================
    print("\n" + "=" * 60)
    print("Setting A: Train PPO on REAL environment (baseline)")
    print("=" * 60)

    real_env = make_vec_env(
        lambda: Quadrotor2DEnv(randomize_params=True), n_envs=4
    )
    model_real = PPO("MlpPolicy", real_env, verbose=0,
                     learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99,
                     policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    model_real.learn(total_timesteps=args.timesteps, progress_bar=True)
    model_real.save("results/ppo_real")
    real_env.close()

    # Evaluate A in real env
    eval_env_real = Quadrotor2DEnv(randomize_params=False)
    rewards_a, lengths_a = evaluate_agent(model_real, eval_env_real)
    results["A: Real→Real"] = (rewards_a, lengths_a)
    print(f"  Mean reward: {rewards_a.mean():.2f} ± {rewards_a.std():.2f}")
    print(f"  Mean length: {lengths_a.mean():.1f} ± {lengths_a.std():.1f}")
    eval_env_real.close()

    # ============================================================
    # Setting B: Train on SURROGATE env
    # ============================================================
    print("\n" + "=" * 60)
    print("Setting B: Train PPO on SURROGATE environment")
    print("=" * 60)

    surr_env = make_vec_env(lambda: SurrogateEnv(), n_envs=4)
    model_surr = PPO("MlpPolicy", surr_env, verbose=0,
                     learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99,
                     policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    model_surr.learn(total_timesteps=args.timesteps, progress_bar=True)
    model_surr.save("results/ppo_surrogate")
    surr_env.close()

    # Evaluate B in surrogate env
    eval_env_surr = SurrogateEnv()
    rewards_b, lengths_b = evaluate_agent(model_surr, eval_env_surr)
    results["B: Surr→Surr"] = (rewards_b, lengths_b)
    print(f"  Mean reward: {rewards_b.mean():.2f} ± {rewards_b.std():.2f}")
    print(f"  Mean length: {lengths_b.mean():.1f} ± {lengths_b.std():.1f}")
    eval_env_surr.close()

    # ============================================================
    # Setting C: Train on SURROGATE, test on REAL (transfer test)
    # ============================================================
    print("\n" + "=" * 60)
    print("Setting C: Train on Surrogate → Test on REAL (transfer)")
    print("=" * 60)

    eval_env_real2 = Quadrotor2DEnv(randomize_params=False)
    rewards_c, lengths_c = evaluate_agent(model_surr, eval_env_real2)
    results["C: Surr→Real"] = (rewards_c, lengths_c)
    print(f"  Mean reward: {rewards_c.mean():.2f} ± {rewards_c.std():.2f}")
    print(f"  Mean length: {lengths_c.mean():.1f} ± {lengths_c.std():.1f}")
    eval_env_real2.close()

    # ============================================================
    # Visualization
    # ============================================================
    plot_comparison(results)
    print_summary(results)


def plot_comparison(results):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Reward comparison ---
    ax = axes[0]
    labels = list(results.keys())
    reward_data = [results[k][0] for k in labels]
    bp = ax.boxplot(reward_data, labels=labels, patch_artist=True)
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Comparison')
    ax.grid(True, alpha=0.3)
    # Add mean values as text
    for i, (label, (rewards, _)) in enumerate(results.items()):
        ax.text(i + 1, rewards.mean(), f'{rewards.mean():.1f}',
                ha='center', va='bottom', fontweight='bold')

    # --- Episode length comparison ---
    ax = axes[1]
    length_data = [results[k][1] for k in labels]
    bp = ax.boxplot(length_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length Comparison')
    ax.grid(True, alpha=0.3)
    for i, (label, (_, lengths)) in enumerate(results.items()):
        ax.text(i + 1, lengths.mean(), f'{lengths.mean():.1f}',
                ha='center', va='bottom', fontweight='bold')

    fig.suptitle('Real Environment vs. Surrogate Environment', fontsize=14)
    fig.tight_layout()
    fig.savefig('results/comparison.png', dpi=150)
    plt.close(fig)
    print("\nComparison plot saved to results/comparison.png")


def print_summary(results):
    """
    Print the final analysis.

    === THINK ABOUT THIS (final reflection) ===
    Q: Look at the numbers. What do they tell you?

    If C ≈ A: Great! The surrogate is a faithful stand-in for reality.
              You could collect data from a REAL drone, train a surrogate,
              and develop RL controllers entirely in simulation.

    If C << A: The surrogate has errors that matter. Next steps:
              - Collect more/better data
              - Use a bigger/better surrogate architecture
              - Use ensemble methods for uncertainty quantification
              - Mix surrogate + real experience (Dyna framework)
    """
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Setting':<20} {'Reward (mean±std)':<25} {'Length (mean±std)'}")
    print("-" * 60)
    for label, (rewards, lengths) in results.items():
        print(f"{label:<20} {rewards.mean():>7.2f} ± {rewards.std():<7.2f} "
              f"{lengths.mean():>7.1f} ± {lengths.std():.1f}")

    # Transfer gap analysis
    r_real = results["A: Real→Real"][0].mean()
    r_transfer = results["C: Surr→Real"][0].mean()
    gap = abs(r_real - r_transfer) / (abs(r_real) + 1e-8) * 100
    print(f"\nTransfer gap (A vs C): {gap:.1f}%")
    if gap < 20:
        print("-> Excellent transfer! Surrogate is a good approximation.")
    elif gap < 50:
        print("-> Moderate transfer. Surrogate captures main dynamics.")
    else:
        print("-> Large gap. Surrogate needs improvement (more data, bigger model).")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200000)
    args = parser.parse_args()
    train_and_evaluate(args)
