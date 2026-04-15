"""
Surrogate-Based Gymnasium Environment
=======================================
This environment replaces the physics simulator with our learned surrogate.

Instead of:  next_state = physics_equations(state, action, mass, inertia, ...)
We use:      next_state = neural_network(state, action)

=== THIS IS THE KEY IDEA ===

If the surrogate is good enough, we can:
  1. Train RL agents WITHOUT a physics simulator (just the neural network)
  2. Train MUCH FASTER (NN forward pass vs. physics integration)
  3. Handle real UAVs where physics are unknown (learn from flight data)

=== THINK ABOUT THIS ===
Q1: What are the risks of training RL on a surrogate?
    -> "Model exploitation": the RL agent finds states where the surrogate
       is inaccurate and exploits them for artificial high reward. The
       agent looks great in the surrogate world but fails in reality.
       This is a MAJOR challenge in model-based RL.

Q2: How could you mitigate this?
    -> Ensembles: train multiple surrogates, penalize disagreement
    -> Dyna-style: mix real and surrogate experience
    -> Conservative estimates: be pessimistic about uncertain states
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from models.surrogate import SurrogateModel


class SurrogateEnv(gym.Env):
    """
    Gymnasium env that uses a learned neural network as its dynamics.
    Same interface as Quadrotor2DEnv, but no physics equations inside.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, surrogate_path="results/surrogate_final.pth", render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Load the trained surrogate
        self.surrogate = SurrogateModel(state_dim=6, action_dim=2)
        self.surrogate.load_state_dict(
            torch.load(surrogate_path, weights_only=True)
        )
        self.surrogate.eval()

        # Same spaces as the real environment
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_high = np.array([5, 5, np.pi, 10, 10, 10, 5, 5], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.max_steps = 300

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Identical initialization to Quadrotor2DEnv
        self.state = np.array([
            self.np_random.uniform(-0.3, 0.3),
            self.np_random.uniform(0.5, 1.5),
            0.0,
            0.0, 0.0, 0.0
        ], dtype=np.float64)

        self.target = np.array([
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(0.5, 3.5),
        ], dtype=np.float64)

        self.step_count = 0
        self._prev_distance = np.linalg.norm(self.state[:2] - self.target)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.state, self.target]).astype(np.float32)

    @torch.no_grad()
    def step(self, action):
        """
        Use the surrogate network instead of physics equations.

        === COMPARE WITH Quadrotor2DEnv.step() ===
        Real env:     next_state = Euler_integration(F=ma, torque=I*alpha, ...)
        Surrogate:    next_state = neural_network.predict(state, action)

        The RL agent can't tell the difference — it just sees
        (observation, reward, done) either way.
        """
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        state_tensor = torch.tensor(self.state[:6], dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        # Neural network predicts next state
        next_state = self.surrogate.predict_next_state(state_tensor, action_tensor)
        self.state = next_state.squeeze(0).numpy().astype(np.float64)

        # Wrap angle
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi

        self.step_count += 1

        # Same reward function as real env
        reward = self._compute_reward()

        out_of_bounds = abs(self.state[0]) > 5 or abs(self.state[1]) > 5
        hit_ground = self.state[1] < -0.5
        flipped = abs(self.state[2]) > 1.2
        timed_out = self.step_count >= self.max_steps

        terminated = out_of_bounds or flipped or hit_ground
        truncated = timed_out

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_reward(self):
        """Identical reward to the real environment."""
        pos = self.state[:2]
        theta = self.state[2]
        omega = self.state[5]
        vel = self.state[3:5]

        distance = np.linalg.norm(pos - self.target)

        progress = self._prev_distance - distance
        self._prev_distance = distance

        reward = (
            10.0 * progress
            - 0.3 * abs(theta)
            - 0.05 * abs(omega)
            - 0.01 * np.linalg.norm(vel)
            + 0.1
        )

        if distance < 0.3:
            reward += 5.0
        elif distance < 0.8:
            reward += 1.0

        if abs(self.state[0]) > 5 or abs(self.state[1]) > 5 or self.state[1] < -0.5:
            reward -= 5.0
        if abs(theta) > 1.2:
            reward -= 5.0

        return reward
