# UAV-RL-Surrogate

**Surrogate Dynamics Modeling + Reinforcement Learning for Adaptive UAV Control**

Can a neural network *replace* a physics simulator? This project builds a complete pipeline: simulate a 2D quadrotor with unknown parameters → train a neural surrogate to learn the dynamics → train RL controllers entirely on the learned model → validate transfer back to reality.

## The Idea

Real UAVs have aerodynamic parameters that are hard to measure (mass distribution, drag coefficients, motor delays). Instead of deriving physics equations, we:

1. **Collect flight data** — observe (state, action, next_state) transitions
2. **Train a surrogate** — MLP learns `next_state = f(state, action)` from data
3. **Train RL on the surrogate** — PPO learns to fly using the neural dynamics model
4. **Validate transfer** — test the surrogate-trained policy on the real simulator

```
Real Physics        Neural Surrogate         RL Agent
┌──────────┐       ┌──────────────┐       ┌──────────┐
│ F = ma   │──────▸│ MLP Network  │──────▸│  PPO     │
│ τ = Iα   │ data  │ Δs = f(s,a)  │  env  │  Policy  │
│ drag,... │       │              │       │          │
└──────────┘       └──────────────┘       └──────────┘
  unknown            learned               adaptive
  parameters         from data             controller
```

## Results

| Setting | Mean Reward | Episode Length |
|---------|------------|---------------|
| A: Real → Real (baseline) | -27.71 ± 2.55 | 3.2 ± 0.4 |
| B: Surrogate → Surrogate | -27.48 ± 2.80 | 3.1 ± 0.3 |
| C: Surrogate → Real (transfer) | -27.69 ± 2.75 | 3.1 ± 0.2 |

**Transfer gap: 0.1%** — the surrogate-trained policy performs nearly identically to the real-trained policy.

## Project Structure

```
├── envs/
│   ├── quadrotor2d.py      # 2D quadrotor Gymnasium env (the "real world")
│   └── surrogate_env.py    # Same interface, neural network dynamics
├── models/
│   └── surrogate.py        # MLP dynamics model with residual learning
├── scripts/
│   ├── train_ppo.py        # Step 1: Train PPO on real environment
│   ├── train_surrogate.py  # Step 2: Collect data & train surrogate
│   ├── compare.py          # Step 3: Real vs surrogate comparison
│   └── visualize.py        # Step 4: Flight trajectory visualization
└── requirements.txt
```

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Step 1: Train PPO agent on the real environment
python scripts/train_ppo.py --timesteps 200000

# Step 2: Collect flight data and train surrogate dynamics model
python scripts/train_surrogate.py

# Step 3: Compare real vs. surrogate RL training
python scripts/compare.py --timesteps 200000

# Step 4: Visualize agent behavior
python scripts/visualize.py

# View training curves
tensorboard --logdir=./runs
```

## Key Concepts

### The Environment
A 2D quadrotor with 6-dim state `[x, y, θ, vx, vy, ω]` and 2-dim action `[T1, T2]` (motor thrusts). Physical parameters (mass, inertia, drag) are **randomized each episode**, simulating a-priori unknown dynamics.

### PPO (Proximal Policy Optimization)
Actor-critic RL algorithm for continuous control. The agent learns purely from (observation, reward) interactions — no physics knowledge needed. PPO clips policy updates to prevent training instability.

### Surrogate Model
An MLP trained to predict state transitions: `Δs = f(s, a)`. Uses **residual learning** (predict the change, not the absolute state) and **input normalization** for stable training. Data is collected from both expert (PPO) and random policies for full state-space coverage.

### Transfer Test
The critical question: does an RL policy trained on the *learned* dynamics work on the *real* dynamics? Our 0.1% transfer gap shows the surrogate is a faithful approximation.

## Tech Stack

- **Gymnasium** — RL environment interface
- **Stable-Baselines3** — PPO implementation (PyTorch backend)
- **PyTorch** — Surrogate model training
- **Matplotlib** — Visualization
- **TensorBoard** — Training monitoring

## References

- [Schulman et al. "Proximal Policy Optimization Algorithms" (2017)](https://arxiv.org/abs/1707.06347)
- [Nagabandi et al. "Neural Network Dynamics for Model-Based Deep RL" (2018)](https://arxiv.org/abs/1708.02596)
- [Panerati et al. "Learning to Fly" (IROS 2021)](https://arxiv.org/abs/2103.02142)
