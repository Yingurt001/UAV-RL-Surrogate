"""
2D Quadrotor Gymnasium Environment
===================================
This is the "real world" simulator. The quadrotor has physical parameters
(mass, inertia, arm length, drag) that we treat as UNKNOWN to the agent.

The RL agent only sees observations and rewards — it never peeks at these
parameters. Later, we'll train a surrogate neural network to *learn* these
dynamics from data, replacing this simulator entirely.

                     T1        T2
                      ^        ^
                      |        |
                 +----|--------|----+
                 |    L    CG    L  |
                 +------------------+
                         |
                         v  mg (gravity)

State:  [x, y, theta, vx, vy, omega]  (6-dim)
Action: [T1, T2]  — thrust of left/right motor, normalized to [0, 1]

=== THINK ABOUT THIS ===
Q1: Why do we separate "environment" from "agent"?
    -> Because in real UAVs, you can't access the physics equations either.
       The agent must learn purely from interaction (state, action, reward).

Q2: Why randomize physical parameters?
    -> This simulates "a-priori unknown parameters" — exactly what your
       UAV research was about. The agent (or surrogate) must generalize
       across different UAV configurations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Quadrotor2DEnv(gym.Env):
    """
    A 2D quadrotor that must reach a target position.

    The key insight: physical parameters are randomized each episode,
    so the controller cannot rely on knowing exact dynamics.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, randomize_params=True):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_params = randomize_params

        # --- Action space ---
        # Two motors, each outputting normalized thrust [0, 1]
        # Think: why normalize? -> Keeps the RL algorithm's output range stable
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- Observation space ---
        # [x, y, theta, vx, vy, omega, target_x, target_y] = 8-dim
        obs_high = np.array([5, 5, np.pi, 10, 10, 10, 5, 5], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # --- Simulation ---
        self.dt = 0.02       # 50 Hz — typical drone control frequency
        self.max_steps = 300  # 6 seconds per episode
        self.max_thrust = 6.0  # Newtons per motor (hover ≈ 0.41 normalized)

        # --- Default physical parameters (will be randomized) ---
        self._default_params = {
            "mass": 0.5,        # kg
            "inertia": 0.025,   # kg*m^2 — high enough to resist flipping
            "arm_length": 0.2,  # m — distance from center to motor
            "drag": 0.2,        # aerodynamic drag — helps natural stabilization
        }
        self.gravity = 9.81

        # Rendering
        self._screen = None
        self._clock = None

    def _randomize_physics(self):
        """
        Randomize physical parameters within ±20% of default.

        === THINK ABOUT THIS ===
        Q3: What happens if we DON'T randomize?
            -> The agent overfits to one specific UAV. Put it on a slightly
               heavier drone and it crashes. This is called "sim-to-real gap".
        """
        if self.randomize_params:
            scale = lambda v: v * self.np_random.uniform(0.8, 1.2)
            self.mass = scale(self._default_params["mass"])
            self.inertia = scale(self._default_params["inertia"])
            self.arm_length = scale(self._default_params["arm_length"])
            self.drag = scale(self._default_params["drag"])
        else:
            self.mass = self._default_params["mass"]
            self.inertia = self._default_params["inertia"]
            self.arm_length = self._default_params["arm_length"]
            self.drag = self._default_params["drag"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._randomize_physics()

        # Start near origin with small random perturbation
        self.state = np.array([
            self.np_random.uniform(-0.3, 0.3),  # x
            self.np_random.uniform(0.5, 1.5),    # y — start above ground
            0.0,                                  # theta — start level
            0.0, 0.0, 0.0                        # zero initial velocity
        ], dtype=np.float64)

        # Random target — reachable distance
        self.target = np.array([
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(0.5, 3.5),  # target always above ground
        ], dtype=np.float64)

        self.step_count = 0
        self._prev_distance = np.linalg.norm(self.state[:2] - self.target)
        return self._get_obs(), {}

    def _get_obs(self):
        """Concatenate state + target into observation vector."""
        return np.concatenate([self.state, self.target]).astype(np.float32)

    def step(self, action):
        """
        Apply motor thrusts and simulate one timestep of physics.

        === THE PHYSICS (this is what the surrogate will learn) ===

        Forces on the quadrotor:
          - Each motor produces vertical thrust (in body frame)
          - Gravity pulls down (world frame)
          - Drag opposes velocity (world frame)

        The rotation matrix converts body-frame forces to world-frame:
          F_world = R(theta) @ F_body

        === THINK ABOUT THIS ===
        Q4: Why can't we just use a PID controller here?
            -> PID needs to know (or tune for) the plant model. When mass
               and inertia are unknown and change each episode, PID fails.
               RL learns a *policy* that adapts from observation alone.
        """
        action = np.clip(action, 0.0, 1.0)
        T1 = action[0] * self.max_thrust  # left motor
        T2 = action[1] * self.max_thrust  # right motor

        x, y, theta, vx, vy, omega = self.state

        # Total thrust and torque
        total_thrust = T1 + T2
        # Torque = (T2 - T1) * arm_length — differential thrust causes rotation
        torque = (T2 - T1) * self.arm_length

        # Body-to-world rotation for thrust direction
        # Think: thrust pushes "up" relative to the drone, not relative to ground
        ax = -total_thrust * np.sin(theta) / self.mass - self.drag * vx / self.mass
        ay = total_thrust * np.cos(theta) / self.mass - self.gravity - self.drag * vy / self.mass
        # Angular drag to help stabilize rotation
        alpha = torque / self.inertia - 0.5 * omega

        # Semi-implicit Euler integration
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        omega_new = omega + alpha * self.dt
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        theta_new = theta + omega_new * self.dt

        # Wrap angle to [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        self.state = np.array([x_new, y_new, theta_new, vx_new, vy_new, omega_new])
        self.step_count += 1

        # --- Reward design ---
        reward = self._compute_reward()

        # --- Termination ---
        out_of_bounds = abs(x_new) > 5 or abs(y_new) > 5
        hit_ground = y_new < -0.5
        flipped = abs(theta_new) > 1.2  # ~69 degrees, more forgiving than pi/2
        timed_out = self.step_count >= self.max_steps

        terminated = out_of_bounds or flipped or hit_ground
        truncated = timed_out

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_reward(self):
        """
        Reward = how well are we doing?

        === THINK ABOUT THIS ===
        Q5: Why not just reward = -distance? Why add angle and velocity penalties?
            -> Pure distance reward: the drone might learn to fly fast but crash.
               We want it to reach the target AND stay stable AND not waste energy.
               This is called "reward shaping" — guiding the agent toward good behavior.

        Q6: What is "reward shaping" with a progress term?
            -> Instead of just penalizing current distance, we reward PROGRESS:
               if distance decreased since last step, that's positive reward.
               This gives the agent a clearer signal: "you're going the right way."
        """
        pos = self.state[:2]
        theta = self.state[2]
        omega = self.state[5]
        vel = self.state[3:5]

        distance = np.linalg.norm(pos - self.target)

        # Progress reward — did we get closer?
        progress = self._prev_distance - distance
        self._prev_distance = distance

        reward = (
            10.0 * progress               # strong signal for approaching target
            - 0.3 * abs(theta)             # stay upright
            - 0.05 * abs(omega)            # smooth rotation
            - 0.01 * np.linalg.norm(vel)   # don't go too fast
            + 0.1                          # survival bonus: alive = good
        )

        # Big bonus for reaching target
        if distance < 0.3:
            reward += 5.0
        elif distance < 0.8:
            reward += 1.0

        # Crash penalties
        if abs(self.state[0]) > 5 or abs(self.state[1]) > 5 or self.state[1] < -0.5:
            reward -= 5.0
        if abs(theta) > 1.2:
            reward -= 5.0

        return reward

    def render(self):
        """Render the quadrotor and target using matplotlib (simple version)."""
        if self.render_mode is None:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not hasattr(self, '_fig'):
            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1, figsize=(6, 6))

        ax = self._ax
        ax.clear()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Step {self.step_count}')

        # Draw ground
        ax.axhline(y=-0.5, color='brown', linewidth=2)
        ax.fill_between([-5, 5], -1, -0.5, color='burlywood', alpha=0.3)

        # Draw target
        ax.plot(*self.target, 'r*', markersize=15, label='Target')

        # Draw drone body
        x, y, theta = self.state[:3]
        L = self.arm_length * 3  # scale for visibility
        dx = L * np.cos(theta)
        dy = L * np.sin(theta)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], 'b-', linewidth=3)
        ax.plot(x, y, 'ko', markersize=5)

        ax.legend()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def close(self):
        if hasattr(self, '_fig'):
            import matplotlib.pyplot as plt
            plt.close(self._fig)
