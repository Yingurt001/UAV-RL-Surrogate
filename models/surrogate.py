"""
Surrogate Dynamics Model
=========================
This neural network learns to predict: next_state = f(state, action)

Instead of coding physics equations (F=ma, torque=I*alpha, etc.), we
let a neural network LEARN the dynamics from data. This is exactly what
"surrogate modeling for a-priori unknown parameters" means on your CV.

=== WHY IS THIS USEFUL? ===

Real UAVs have complex aerodynamics that are hard to model analytically:
  - Blade flapping, ground effect, turbulence, motor delays...
  - Physical parameters change over time (battery drains, payload shifts)

A surrogate model can:
  1. Be trained from flight data (no physics knowledge needed)
  2. Adapt to new conditions by retraining on new data
  3. Run faster than physics simulators (good for planning/MPC)

=== THINK ABOUT THIS ===
Q1: Why predict (next_state - current_state) instead of next_state directly?
    -> This is called "residual learning". The network only needs to learn
       the CHANGE (which is small), not the absolute state (which varies wildly).
       This makes training much easier and more accurate.

Q2: What happens if the surrogate is inaccurate?
    -> The RL agent trained on an inaccurate surrogate will learn a bad
       policy. This is the "sim-to-real gap" problem. Later we'll measure
       this by comparing real-env vs surrogate-env performance.
"""

import torch
import torch.nn as nn
import numpy as np


class SurrogateModel(nn.Module):
    """
    MLP that predicts state transition: delta_state = f(state, action)

    Architecture: [state_dim + action_dim] -> 128 -> 128 -> [state_dim]

    === THINK ABOUT THIS ===
    Q3: Why 2 hidden layers of 128? Why not deeper?
        -> Our dynamics are relatively simple (6-dim state, 2D physics).
           A shallow network is enough and trains faster. For real 3D UAVs
           with 12+ dim state, you'd want deeper networks (256-256-256).
    """

    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Store normalization stats (computed from training data)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))
        self.register_buffer('delta_mean', torch.zeros(state_dim))
        self.register_buffer('delta_std', torch.ones(state_dim))

    def forward(self, state, action):
        """
        Predict state change given current state and action.

        Input normalization is critical:
          - Raw state: x might be in [-5, 5], theta in [-pi, pi]
          - Without normalization, the network struggles with different scales
        """
        # Normalize inputs
        state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
        action_norm = (action - self.action_mean) / (self.action_std + 1e-8)

        x = torch.cat([state_norm, action_norm], dim=-1)
        delta_norm = self.net(x)

        # Denormalize output
        delta = delta_norm * self.delta_std + self.delta_mean
        return delta

    def predict_next_state(self, state, action):
        """Predict next_state = current_state + predicted_delta."""
        delta = self.forward(state, action)
        return state + delta

    def set_normalization(self, states, actions, deltas):
        """Compute and store normalization statistics from training data."""
        self.state_mean = torch.tensor(states.mean(axis=0), dtype=torch.float32)
        self.state_std = torch.tensor(states.std(axis=0), dtype=torch.float32)
        self.action_mean = torch.tensor(actions.mean(axis=0), dtype=torch.float32)
        self.action_std = torch.tensor(actions.std(axis=0), dtype=torch.float32)
        self.delta_mean = torch.tensor(deltas.mean(axis=0), dtype=torch.float32)
        self.delta_std = torch.tensor(deltas.std(axis=0), dtype=torch.float32)
