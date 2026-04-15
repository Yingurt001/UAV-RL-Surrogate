"""
Step 1: Train PPO on the Real Environment
==========================================
This script trains a PPO (Proximal Policy Optimization) agent to control
the 2D quadrotor. The agent learns ONLY from (observation, reward) — it
never sees the physics equations or parameters.

=== WHAT IS PPO? (the intuition) ===

PPO is an "actor-critic" algorithm:
  - Actor:  a neural network that outputs actions (motor thrusts)
  - Critic: a neural network that estimates "how good is this state?"

The training loop:
  1. Collect a batch of experience (rollouts) using current policy
  2. Compute advantages: "was this action better or worse than expected?"
  3. Update the actor to do more of what worked, less of what didn't
  4. Update the critic to better predict future rewards
  5. CLIP the update to prevent too-large policy changes (the "proximal" part)

=== THINK ABOUT THIS ===
Q1: Why PPO and not DQN?
    -> DQN is for DISCRETE actions (left/right/up/down).
       Our motors output CONTINUOUS thrust values. PPO handles continuous
       action spaces natively.

Q2: What does "proximal" mean?
    -> It means "nearby". PPO constrains each update to stay close to the
       previous policy. Big jumps in policy = training instability.
       The clip ratio (default 0.2) controls how far the policy can move.

Q3: Why use a separate critic (value function)?
    -> Without a critic, you'd need to wait until episode end to compute
       returns. The critic estimates future return at each step, enabling
       learning from incomplete episodes (bootstrapping).

Usage:
    python scripts/train_ppo.py
    python scripts/train_ppo.py --timesteps 500000  # longer training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from envs.quadrotor2d import Quadrotor2DEnv


def make_env(randomize=True):
    """Factory function for creating environments."""
    def _init():
        return Quadrotor2DEnv(randomize_params=randomize)
    return _init


def train(args):
    """
    Train PPO agent.

    === KEY HYPERPARAMETERS (play with these!) ===

    learning_rate: How fast the network updates weights.
        Too high -> unstable, too low -> slow convergence.
        Try: 1e-4 to 1e-3

    n_steps: How many steps to collect before each update.
        More steps = less variance but slower updates.
        Try: 1024 to 4096

    batch_size: Mini-batch size for SGD.
        Must divide n_steps * n_envs evenly.

    n_epochs: How many times to reuse each batch of experience.
        More epochs = more sample-efficient but risk overfitting.

    gamma: Discount factor. How much to value future rewards.
        0.99 = long-horizon, 0.9 = short-sighted.

    clip_range: The PPO clipping parameter (epsilon).
        0.2 is standard. Smaller = more conservative updates.

    === THINK ABOUT THIS ===
    Q4: We use 4 parallel environments (n_envs=4). Why?
        -> Each env has different randomized params. The agent sees diverse
           dynamics simultaneously, learning a more robust policy. This is
           like training on 4 different drones at once.
    """
    # Create vectorized environments (4 in parallel for diversity)
    n_envs = 4
    train_env = make_vec_env(make_env(randomize=True), n_envs=n_envs)

    # Evaluation env with FIXED params — measures true performance
    eval_env = make_vec_env(make_env(randomize=False), n_envs=1)

    # PPO with MLP policy (two hidden layers of 64 units)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,          # steps per env before update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,         # entropy bonus — encourages exploration
        verbose=1,
        tensorboard_log="./runs/ppo_quadrotor",
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])  # actor and critic networks
        ),
    )

    # Callback: evaluate every 10k steps, save best model
    os.makedirs("results", exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./results/",
        log_path="./results/",
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"\n{'='*60}")
    print(f"Training PPO for {args.timesteps:,} timesteps")
    print(f"  n_envs={n_envs}, n_steps=2048, batch_size=64")
    print(f"  Learning rate: 3e-4, Gamma: 0.99, Clip: 0.2")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("results/ppo_quadrotor_final")
    print(f"\nModel saved to results/ppo_quadrotor_final.zip")
    print(f"Best model saved to results/best_model.zip")
    print(f"\nTo view training curves:")
    print(f"  tensorboard --logdir=./runs")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on 2D Quadrotor")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps (default: 200000)")
    args = parser.parse_args()
    train(args)
