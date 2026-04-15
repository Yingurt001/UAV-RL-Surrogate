# UAV Surrogate + RL 实战教程

> 从零搭建：用神经网络学习无人机动力学，用强化学习训练飞行控制器

本教程对应你简历上的 UAV 研究经历，完成后你将完全理解以下三件事：
- **Surrogate Modeling**: 用 MLP 替代物理方程预测无人机状态
- **PPO 强化学习**: Actor-Critic 算法如何学会连续控制
- **Sim-to-Real Transfer**: 在"假世界"训练的策略能否在"真世界"工作

---

## 0. 环境准备

```bash
cd "/Users/zhangying/Personal/Andrea/Academic 学术/Research 研究/UAV project"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

依赖很轻量：`stable-baselines3`（PPO）、`gymnasium`（RL 接口）、`torch`（神经网络）、`matplotlib`（画图）。

---

## 1. 理解环境：2D 四旋翼

> **核心文件**: `envs/quadrotor2d.py`

### 1.1 先跑起来看看

打开 Python，手动操控无人机感受一下：

```python
from envs.quadrotor2d import Quadrotor2DEnv
import numpy as np

env = Quadrotor2DEnv(randomize_params=False)
obs, _ = env.reset(seed=42)

print(f"观测空间: {env.observation_space.shape}")  # (8,)
print(f"动作空间: {env.action_space.shape}")        # (2,)
print(f"初始观测: {obs}")
# obs = [x, y, theta, vx, vy, omega, target_x, target_y]
```

**停下来想**：观测有 8 维，但无人机状态只有 6 维（位置+角度+速度）。多出的 2 维是什么？为什么要把 target 放进观测？

### 1.2 理解动力学

无人机有两个电机，产生推力 T1（左）和 T2（右）：

```
     T1        T2
      ↑         ↑
      |         |
 +----|----CG----|----+
      ← L →  ← L →
              |
              ↓ mg
```

物理公式（这些就是 surrogate 要学的东西）：

```
水平加速度:  ax = -(T1+T2) * sin(θ) / m - drag * vx / m
垂直加速度:  ay = (T1+T2) * cos(θ) / m - g - drag * vy / m
角加速度:    α  = (T2-T1) * L / I - 0.5 * ω
```

**停下来想**：
- 为什么推力要乘以 sin(θ) 和 cos(θ)？（提示：推力方向相对于机身，不是地面）
- 如果两个电机推力相同 (T1=T2)，力矩是多少？无人机会转吗？
- 悬停需要多大的推力？（提示：hover 时 T1+T2 = mg，代入数字算一下）

### 1.3 手动控制

试试用固定推力让无人机悬停：

```python
# 计算悬停推力
mass = 0.5  # kg
g = 9.81
max_thrust = 6.0  # N per motor
hover_thrust = mass * g / (2 * max_thrust)  # 归一化推力
print(f"悬停推力 (归一化): {hover_thrust:.3f}")  # ≈ 0.409

# 让无人机悬停 200 步
obs, _ = env.reset(seed=0)
for step in range(200):
    action = np.array([hover_thrust, hover_thrust])
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated:
        print(f"Step {step}: 坠毁! theta={obs[2]:.3f}")
        break
else:
    print(f"悬停成功 200 步! y={obs[1]:.3f}, theta={obs[2]:.4f}")
```

**停下来想**：悬停成功了，但无人机能飞到目标吗？要飞到右边 (target_x > 0)，应该怎么调整 T1 和 T2？

### 1.4 参数随机化

这是你简历上 "a-priori unknown parameters" 的体现：

```python
# 每次 reset，物理参数都不同
env_rand = Quadrotor2DEnv(randomize_params=True)
for i in range(5):
    obs, _ = env_rand.reset()
    print(f"Episode {i}: mass={env_rand.mass:.3f}, inertia={env_rand.inertia:.4f}, "
          f"arm={env_rand.arm_length:.3f}, drag={env_rand.drag:.3f}")
```

**停下来想**：如果你用 PID 控制器，参数变了就得重新调参。RL 呢？它从观测学策略，不依赖已知参数——这就是 RL 的优势。

### 1.5 Reward 设计

看一下 reward 函数怎么引导 agent：

```python
obs, _ = env.reset(seed=0)
target = obs[6:8]
print(f"目标位置: ({target[0]:.2f}, {target[1]:.2f})")

# 随机动作 vs 悬停动作
for name, action in [("随机", env.action_space.sample()),
                      ("悬停", np.array([hover_thrust, hover_thrust]))]:
    obs_test, _ = env.reset(seed=0)
    _, reward, _, _, _ = env.step(action)
    print(f"{name}动作 -> reward = {reward:.3f}")
```

Reward 有四个部分：
1. **进步奖励** `10 * (prev_dist - curr_dist)`: 靠近目标 = 正奖励
2. **姿态惩罚** `-0.3 * |θ|`: 歪了就扣分
3. **旋转惩罚** `-0.05 * |ω|`: 转太快扣分
4. **存活奖励** `+0.1`: 活着就有分

**停下来想**：为什么"进步奖励"比"绝对距离惩罚"好？（提示：想象无人机在远处，每步 reward 都是 -5.0，它怎么知道该往哪飞？）

---

## 2. 训练 PPO 强化学习

> **核心文件**: `scripts/train_ppo.py`

### 2.1 PPO 是什么？(30 秒版本)

PPO 有两个神经网络：
- **Actor (策略网络)**: 输入观测 → 输出动作（两个电机推力）
- **Critic (价值网络)**: 输入观测 → 输出"这个状态有多好"的估计

训练循环：
1. 用当前策略收集一批经验 (state, action, reward, next_state)
2. 算 advantage：这个动作比预期好多少？
3. 更新 Actor：多做 advantage > 0 的动作，少做 advantage < 0 的
4. 更新 Critic：让它更准地预测未来 reward
5. **Clip**: 限制更新幅度，防止策略跳太远（PPO 的核心）

### 2.2 开始训练

```bash
python scripts/train_ppo.py --timesteps 500000
```

训练过程中观察这些指标：
- `ep_rew_mean`: 平均 episode reward，应该从负数涨到 1300+
- `ep_len_mean`: 平均 episode 长度，应该从 ~50 涨到 300（满 episode）
- `entropy_loss`: 策略的随机性，应该逐渐下降（从探索到利用）
- `clip_fraction`: 被 clip 的比例，应该在 0.1-0.2（太高说明更新太激进）

**停下来想**：
- 为什么用 4 个并行环境 (`n_envs=4`)？（提示：每个环境参数不同）
- `gamma=0.99` 是什么？改成 0.9 会怎样？（提示：gamma 越小，agent 越"短视"）

### 2.3 查看训练曲线

```bash
tensorboard --logdir=./runs
# 浏览器打开 http://localhost:6006
```

或者直接看评估结果：

```python
from stable_baselines3 import PPO
from envs.quadrotor2d import Quadrotor2DEnv
import numpy as np

model = PPO.load("results/best_model")
env = Quadrotor2DEnv(randomize_params=False)

for ep in range(5):
    obs, _ = env.reset(seed=ep)
    target = obs[6:8]
    total_reward = 0
    steps = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    dist = np.linalg.norm(obs[:2] - target)
    print(f"Ep{ep}: reward={total_reward:.0f}, steps={steps}, "
          f"最终距离目标={dist:.3f}m, theta={obs[2]:.4f}")
```

你应该看到：每个 episode 跑满 300 步，reward ~1300+，最终距离 < 0.1m，theta ≈ 0。

---

## 3. 训练 Surrogate 动力学模型

> **核心文件**: `models/surrogate.py`, `scripts/train_surrogate.py`

这一步是项目的灵魂——用神经网络学习物理定律。

### 3.1 为什么需要 Surrogate？

真实 UAV 的情况：
- 你不知道精确的质量、惯量、阻力系数
- 气动效应复杂（地面效应、桨叶挥舞、湍流）
- 参数会变（电池放电、载荷变化）

Surrogate 的做法：不管物理方程是什么，直接从数据学：
```
输入:  (当前状态 s, 动作 a)
输出:  状态变化 Δs = s' - s
预测:  下一状态 s' = s + Δs
```

### 3.2 为什么预测 Δs 而不是 s'？

这叫 **residual learning**，是最关键的 trick：

```python
# 假设 state = [2.000, 3.000, 0.100, ...]
# 真实 next_state = [2.003, 2.998, 0.095, ...]

# 方法 A: 直接预测 next_state
#   网络要学会输出 [2.003, 2.998, 0.095]
#   数值很大，精度要求高

# 方法 B: 预测 delta (residual)
#   网络只需输出 [0.003, -0.002, -0.005]
#   数值很小，容易学
```

**停下来想**：如果你用方法 A，100 步后误差会累积多少？用方法 B 呢？

### 3.3 数据收集策略

```bash
python scripts/train_surrogate.py
```

数据来自两个来源：
- **50% 专家数据**（训练好的 PPO agent）：覆盖"好"状态
- **50% 随机数据**（随机动作）：覆盖"坏"状态

**停下来想**：为什么不能只用专家数据？（提示：RL 训练初期，agent 会做很多"坏"动作，到达专家没去过的状态。如果 surrogate 在那些状态不准，agent 就学不好。）

### 3.4 查看结果

运行完后会生成两张图：

**surrogate_training_curve.png** — 训练/验证 loss 曲线
- train 和 val loss 应该同步下降
- 如果 val loss 反弹 = 过拟合，需要 early stopping 或更多数据

**surrogate_prediction_quality.png** — 每个维度的预测精度
- 散点图应该紧贴红色对角线
- R² ≈ 1.0 表示预测近乎完美
- 看哪个维度最难预测（通常是 dω，因为角动力学最敏感）

```python
# 手动检查 surrogate 准确度
import torch
from models.surrogate import SurrogateModel
from envs.quadrotor2d import Quadrotor2DEnv
import numpy as np

surrogate = SurrogateModel()
surrogate.load_state_dict(torch.load("results/surrogate_final.pth", weights_only=True))
surrogate.eval()

env = Quadrotor2DEnv(randomize_params=False)
obs, _ = env.reset(seed=0)
action = np.array([0.4, 0.42])  # 稍微不对称的推力

# 真实物理
real_next_obs, _, _, _, _ = env.step(action)
real_next_state = real_next_obs[:6]

# Surrogate 预测
with torch.no_grad():
    s = torch.tensor(obs[:6], dtype=torch.float32).unsqueeze(0)
    a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    pred_next = surrogate.predict_next_state(s, a).squeeze().numpy()

print("真实 next_state:", real_next_state.round(5))
print("预测 next_state:", pred_next.round(5))
print("误差:           ", (real_next_state - pred_next).round(5))
```

---

## 4. 对比实验：真实 vs Surrogate

> **核心文件**: `envs/surrogate_env.py`, `scripts/compare.py`

### 4.1 实验设计

三个 setting，回答一个问题：**在假世界训练的 agent，在真世界能用吗？**

| Setting | 训练环境 | 测试环境 | 回答的问题 |
|---------|---------|---------|-----------|
| A: Real→Real | 物理仿真 | 物理仿真 | baseline，最好能达到多少？ |
| B: Surr→Surr | Surrogate | Surrogate | Surrogate 里能学会飞吗？ |
| C: Surr→Real | Surrogate | 物理仿真 | **关键测试**: 策略能迁移吗？ |

### 4.2 运行实验

```bash
python scripts/compare.py --timesteps 500000
```

这会训练两个独立的 PPO agent（一个在真实环境，一个在 surrogate 环境），然后交叉评估。

### 4.3 解读结果

运行完成后看 `results/comparison.png`，你应该看到：

```
A: Real→Real    1357 ± 48    300 步
B: Surr→Surr    1355 ± 51    300 步
C: Surr→Real    1348 ± 48    300 步
Transfer gap: 0.6%
```

**停下来想**：
- 三个 setting 几乎一样好，说明什么？（surrogate 完美替代了物理仿真）
- 如果 C 远差于 A，该怎么办？（更多数据、更大模型、ensemble、Dyna）
- 这对真实 UAV 意味着什么？（只需要飞行数据，不需要物理方程）

### 4.4 Transfer Gap 的含义

```
                    ┌─────────────┐
  Flight Data ───→  │  Surrogate  │ ───→  RL Agent ───→  Real UAV
  (real drone)      │  (neural net)│      (PPO)           (deployment)
                    └─────────────┘
                         ↑
              如果 transfer gap 小，
              这整条链路就可行！
```

---

## 5. 可视化

> **核心文件**: `scripts/visualize.py`

```bash
python scripts/visualize.py
```

生成三张图：

### flight_trajectories.png
8 条飞行轨迹。圆点=起点，方块=终点，星号=目标。观察：
- 每条轨迹都是平滑的曲线（不是锯齿 → agent 学会了连续控制）
- 终点紧贴目标（方块和星号重合 → 成功到达）

### state_evolution.png
单个 episode 的 6 维状态随时间变化：
- x, y: 快速收敛到 target（红色虚线），说明 agent 学会了导航
- θ: 先倾斜（加速阶段），再回正（减速阶段），最后在 0 附近微调
- vx, vy: 先加速后减速，典型的 bang-bang 控制行为

### action_profile.png
两个电机的推力输出随时间变化。观察：
- 初始阶段 T1≠T2（差分推力产生倾斜，驱动水平运动）
- 稳定后 T1≈T2≈hover（维持悬停在目标位置）

---

## 6. 你学到了什么

完成这个项目，你现在能回答这些面试问题了：

**Q: 你在 UAV 项目中用了什么 ML 算法？**
> 两个层面：(1) MLP 做 surrogate dynamics model，从飞行数据学习状态转移函数 next_state = f(state, action)，使用 residual learning 预测状态变化量；(2) PPO 强化学习做自适应控制，agent 通过 actor-critic 架构学习连续推力控制策略。

**Q: 什么是 surrogate model？为什么需要它？**
> Surrogate 是用神经网络近似复杂物理过程。真实 UAV 的气动参数（质量、惯量、阻力）是未知的且会变化，解析建模困难。Surrogate 从飞行数据直接学习 input-output 映射，不需要知道物理方程。

**Q: PPO 和 DQN 有什么区别？**
> DQN 只能处理离散动作（上下左右），PPO 处理连续动作（电机推力 [0,1]）。PPO 是 on-policy 的 actor-critic 方法，用 clipped surrogate objective 限制每次更新幅度，防止训练不稳定。

**Q: 什么是 sim-to-real transfer？你怎么验证的？**
> 在仿真（或 surrogate）中训练的策略能否在真实环境中工作。我们通过三组对比实验验证：Real→Real 作为 baseline，Surrogate→Real 作为 transfer test。0.6% 的性能差距说明 surrogate 是真实动力学的高保真近似。

**Q: Reward shaping 是什么？为什么重要？**
> 设计 reward 函数引导 agent 学习期望行为。我们用 progress reward（靠近目标的增量）而非 absolute distance（到目标的距离），因为增量信号在远离目标时仍然有方向性，避免了 sparse reward 导致的探索困难。

---

## 7. 进一步探索（可选）

想深入的话，可以尝试：

1. **调超参**: 改 `learning_rate`、`gamma`、`clip_range`，看训练曲线怎么变
2. **更难的任务**: 把目标范围从 ±2m 扩大到 ±4m
3. **Ensemble surrogate**: 训练 5 个 surrogate，用预测方差作为不确定性
4. **Domain randomization 强度**: 把 ±20% 改成 ±50%，看 transfer 是否恶化
5. **3D 扩展**: 用 `gym-pybullet-drones` 做 3D 四旋翼版本

---

## 参考文献

- Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)
- Nagabandi et al. [Neural Network Dynamics for Model-Based Deep RL](https://arxiv.org/abs/1708.02596) (2018)
- Panerati et al. [Learning to Fly — a Gym Environment with PyBullet](https://arxiv.org/abs/2103.02142) (IROS 2021)
