"""
Flappy Bird with Rainbow DQN

Rainbow DQN (Hessel et al., 2018) combines SIX improvements to DQN:

  ┌────────────────────────────────────────────────────────────────────┐
  │  Component              │  What it fixes                          │
  │─────────────────────────│─────────────────────────────────────────│
  │  1. Double DQN          │  Q-value overestimation                 │
  │  2. Dueling DQN         │  Inefficient state/action value sharing │
  │  3. Prioritized Replay  │  Uniform sampling wastes data           │
  │  4. Multi-step (n-step) │  Slow reward propagation                │
  │  5. Distributional (C51)│  Ignoring return distribution shape     │
  │  6. Noisy Networks      │  Crude ε-greedy exploration             │
  └────────────────────────────────────────────────────────────────────┘

Reference: "Rainbow: Combining Improvements in Deep Reinforcement Learning"
           Hessel et al., AAAI 2018

Uses 4-dim state vector [bird_y, velocity_y, pipe_dx, pipe_dy].
"""

import pygame
import random
import sys
import os
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 640
FPS = 60
TRAINING_FPS = 0

# Colors
SKY_TOP = (45, 52, 70)
SKY_BOTTOM = (82, 95, 126)
PIPE_COLOR = (46, 204, 113)
PIPE_HIGHLIGHT = (88, 214, 141)
PIPE_SHADOW = (30, 132, 73)
BIRD_COLOR = (255, 195, 0)
BIRD_EYE = (30, 30, 30)
BIRD_BEAK = (255, 87, 51)
GROUND_COLOR = (139, 90, 43)
GROUND_TOP = (160, 120, 60)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Physics
GRAVITY = 0.5
JUMP_STRENGTH = -9
BIRD_SPEED_X = 3

# Pipe settings
PIPE_WIDTH = 70
PIPE_GAP = 180
PIPE_SPACING = 280
MIN_PIPE_HEIGHT = 80

# Ground
GROUND_HEIGHT = 80

# ─── Rainbow Hyperparameters ─────────────────────────────────
STATE_DIM = 4
HIDDEN_DIM = 128
BATCH_SIZE = 64
MEMORY_SIZE = 100000
GAMMA = 0.99
LEARNING_RATE = 1e-3
TAU = 0.005
TRAIN_EVERY = 4
MIN_MEMORY = 1000

# Rewards
REWARD_ALIVE = 0.1
REWARD_DEAD = -1.0
REWARD_PASS_PIPE = 1.0

# Component 3: Prioritized Experience Replay
PER_ALPHA = 0.6        # Priority exponent (0 = uniform, 1 = full priority)
PER_BETA_START = 0.4   # Importance sampling correction (annealed to 1.0)
PER_BETA_END = 1.0
PER_BETA_ANNEAL = 50000  # Steps to anneal beta from start to end
PER_EPSILON = 1e-6     # Small constant to prevent zero priority

# Component 4: Multi-step returns
N_STEP = 3             # Number of lookahead steps

# Component 5: Distributional RL (C51)
NUM_ATOMS = 51         # Number of atoms in the distribution support
V_MIN = -10.0          # Minimum return value
V_MAX = 10.0           # Maximum return value

# Component 6: Noisy Networks
NOISY_STD = 0.5        # Initial noise standard deviation

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════
#  Component 6: Noisy Linear Layer
# ═══════════════════════════════════════════════════════════════
#
#  Standard Linear:  y = W·x + b              (固定权重)
#  Noisy Linear:     y = (μ + σ⊙ε)·x + (μ_b + σ_b⊙ε_b)
#
#  就是普通 Linear，但权重上叠加了可学习幅度的随机噪声。
#  σ 大 → 噪声大 → 探索多；σ 小 → 噪声小 → 利用多。
#  网络自己通过梯度下降学会调节 σ，取代笨拙的 ε-greedy。

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=NOISY_STD):
        super().__init__()

        # 可学习参数: μ (均值) 和 σ (噪声幅度)
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # ε: 每次前向传播前重新采样的随机噪声 (不是参数，不参与梯度)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        # 初始化
        bound = 1.0 / (in_features ** 0.5)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_init / (in_features ** 0.5))
        self.bias_sigma.data.fill_(sigma_init / (in_features ** 0.5))

        self.reset_noise()

    def reset_noise(self):
        """重新采样噪声 ε ~ N(0,1)，每个训练步调用一次。"""
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            # 训练: weight = μ + σ × ε (带噪声)
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            # 测试: weight = μ (无噪声，确定性)
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


# ═══════════════════════════════════════════════════════════════
#  Component 3: Prioritized Experience Replay (PER)
# ═══════════════════════════════════════════════════════════════
#
#  普通 Replay:     随机均匀采样
#  Prioritized:     按 |TD-error|^α 的比例采样 (错得越多，学得越多)
#
#  但非均匀采样引入偏差 → 用 importance-sampling weight 修正:
#    w_i = (1 / (N × P(i)))^β,  β 从 0.4 退火到 1.0
#
#  实现: 用 numpy 数组存 priority，searchsorted 做 O(log N) 采样。
#  比手写 SumTree 简单得多，numpy 向量化下性能足够。

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.capacity = capacity
        self.alpha = alpha

        self.data = [None] * capacity       # 环形缓冲区存 transition
        self.priorities = np.zeros(capacity, dtype=np.float64)  # 每条的 priority
        self.write_idx = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """新经验给当前最高 priority (假设新数据值得学)。"""
        max_p = self.priorities[:self.size].max() if self.size > 0 else 1.0
        self.data[self.write_idx] = (state, action, reward, next_state, done)
        self.priorities[self.write_idx] = max_p

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        """按 priority 比例采样一个 batch。"""
        # 取有效部分的 priority
        prios = self.priorities[:self.size]

        # 计算采样概率: P(i) = p_i / Σp
        probs = prios / (prios.sum() + 1e-10)

        # 按概率采样索引
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False
                                   if self.size >= batch_size else True)

        # importance-sampling 权重: w = (1/(N·P(i)))^β, 归一化到 max=1
        weights = (self.size * probs[indices] + 1e-10) ** (-beta)
        weights = weights / weights.max()

        # 收集 batch
        batch = [self.data[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device),
            indices,
            torch.tensor(weights, dtype=torch.float32).to(device),
        )

    def update_priorities(self, indices, td_errors):
        """训练后根据新的 TD-error 更新 priority。"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + PER_EPSILON) ** self.alpha

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════
#  Component 4: N-step Return Buffer
# ═══════════════════════════════════════════════════════════════
#
#  Standard 1-step: target = r + γ * V(s')
#  N-step:          target = r₁ + γr₂ + γ²r₃ + γ³V(s''')
#
#  Benefit: reward signal propagates N times faster.
#  Cost: higher variance (more randomness in N-step sum).
#  N=3 is a good balance (from the Rainbow paper).

class NStepBuffer:
    """
    Accumulates N transitions, then emits one (s, a, R_n, s_n, done) tuple
    where R_n = r₁ + γr₂ + γ²r₃ + ... + γ^(n-1)r_n
    """
    def __init__(self, n_step=N_STEP, gamma=GAMMA):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def is_ready(self):
        return len(self.buffer) == self.n_step

    def get(self):
        """
        Compute n-step return and return the aggregated transition.

        R_n = r₁ + γ*r₂ + γ²*r₃
        The state and action come from step 1.
        The next_state comes from step N (or the terminal state).
        """
        state, action = self.buffer[0][0], self.buffer[0][1]

        # Compute discounted n-step return
        R = 0.0
        next_state = self.buffer[-1][3]
        done = False

        for i, (_, _, r, ns, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if d:
                # Episode ended before n steps — truncate
                next_state = ns
                done = True
                break

        return state, action, R, next_state, done

    def flush(self):
        """Flush remaining transitions at episode end (emit shorter-than-n returns)."""
        results = []
        while len(self.buffer) > 0:
            state, action = self.buffer[0][0], self.buffer[0][1]
            R = 0.0
            done = False
            next_state = self.buffer[-1][3]

            for i, (_, _, r, ns, d) in enumerate(self.buffer):
                R += (self.gamma ** i) * r
                if d:
                    next_state = ns
                    done = True
                    break

            results.append((state, action, R, next_state, done))
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()


# ═══════════════════════════════════════════════════════════════
#  Component 5 + 2 + 6: Rainbow Network
#  (Distributional C51 + Dueling architecture + Noisy layers)
# ═══════════════════════════════════════════════════════════════
#
#  Standard DQN:  network outputs Q(s,a) — a single number.
#  C51:           network outputs a DISTRIBUTION over returns.
#
#  Instead of "Q(s, jump) = 5.0", C51 says:
#    "P(return=−10) = 0.01, P(return=−5) = 0.05, ..., P(return=5) = 0.30, ..."
#
#  The support is a fixed grid of NUM_ATOMS values from V_MIN to V_MAX:
#    z = [−10, −9.6, −9.2, ..., 9.6, 10.0]   (51 atoms)
#
#  Q(s,a) = Σ p_i * z_i   (expected value of the distribution)
#
#  Why learn the distribution?
#    - Richer signal than just the mean
#    - More stable gradients (cross-entropy loss on distributions)
#    - Captures risk: same mean but different variance → different behavior

class RainbowNetwork(nn.Module):
    """
    Network architecture:

        state (4)
            │
      ┌─────┴─────┐
      │  Shared    │  (NoisyLinear layers — Component 6)
      │  Features  │
      └─────┬─────┘
            │
      ┌─────┴─────┐
      │           │
    [V stream]  [A stream]       ← Dueling (Component 2)
      │           │
    V(s) dist   A(s,a) dist      ← C51 distributional (Component 5)
    (1×51)      (2×51)
      │           │
      └─────┬─────┘
            │
    Q_dist(s,a) = V + (A - mean(A))   per atom
            │
       softmax → P(s,a)              probability distribution
            │
    Q(s,a) = Σ P_i × z_i            expected value
    """
    def __init__(self, num_atoms=NUM_ATOMS):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_actions = 2

        # Shared feature layers (NoisyLinear for learned exploration)
        self.shared_fc1 = NoisyLinear(STATE_DIM, HIDDEN_DIM)
        self.shared_fc2 = NoisyLinear(HIDDEN_DIM, HIDDEN_DIM)

        # Value stream → outputs distribution over atoms
        self.value_fc = NoisyLinear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.value_out = NoisyLinear(HIDDEN_DIM // 2, num_atoms)  # V distribution

        # Advantage stream → outputs distribution per action
        self.advantage_fc = NoisyLinear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.advantage_out = NoisyLinear(HIDDEN_DIM // 2, self.num_actions * num_atoms)

    def forward(self, x):
        """
        Returns: log-probabilities of shape (batch, num_actions, num_atoms)
        """
        features = F.relu(self.shared_fc1(x))
        features = F.relu(self.shared_fc2(features))

        # Value stream
        v = F.relu(self.value_fc(features))
        v = self.value_out(v)                                     # (batch, atoms)
        v = v.view(-1, 1, self.num_atoms)                         # (batch, 1, atoms)

        # Advantage stream
        a = F.relu(self.advantage_fc(features))
        a = self.advantage_out(a)                                  # (batch, actions*atoms)
        a = a.view(-1, self.num_actions, self.num_atoms)           # (batch, actions, atoms)

        # Dueling combination (per atom): Q = V + A - mean(A)
        q_atoms = v + (a - a.mean(dim=1, keepdim=True))           # (batch, actions, atoms)

        # Convert to log-probabilities (for stable cross-entropy loss)
        log_probs = F.log_softmax(q_atoms, dim=2)                 # (batch, actions, atoms)
        return log_probs

    def get_q_values(self, x, support):
        """Compute expected Q-values: Q(s,a) = Σ p_i * z_i"""
        log_probs = self.forward(x)
        probs = log_probs.exp()                                    # (batch, actions, atoms)
        q_values = (probs * support.unsqueeze(0).unsqueeze(0)).sum(2)  # (batch, actions)
        return q_values

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ═══════════════════════════════════════════════════════════════
#  Rainbow Agent — all 6 components working together
# ═══════════════════════════════════════════════════════════════

class RainbowAgent:
    def __init__(self):
        # C51 support: fixed grid of atom values
        self.support = torch.linspace(V_MIN, V_MAX, NUM_ATOMS).to(device)
        self.delta_z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)

        # Networks (Dueling + Noisy + Distributional)
        self.online_net = RainbowNetwork().to(device)
        self.target_net = RainbowNetwork().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)

        # PER replay buffer
        self.memory = PrioritizedReplayMemory(MEMORY_SIZE)
        self.per_beta = PER_BETA_START

        # N-step buffer
        self.n_step_buffer = NStepBuffer(N_STEP, GAMMA)

        # Statistics
        self.episode = 0
        self.best_score = 0
        self.total_steps = 0

        self.score_history = []
        self.loss_history = []
        self.q_value_history = []
        self.priority_history = []     # Track average priority (PER metric)
        self.epsilon_history = []      # Track noise magnitude (Noisy Net metric)

    def get_action(self, state, training=True):
        """
        Action selection — NO ε-greedy needed!
        Noisy Networks handle exploration automatically.
        During testing, noise is disabled (deterministic).
        """
        self.online_net.train(training)
        with torch.no_grad():
            state_t = state.unsqueeze(0).to(device)
            q_values = self.online_net.get_q_values(state_t, self.support)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        """
        One training step combining ALL 6 components:

        1. Sample prioritized batch (PER)
        2. Compute n-step distributional target (n-step + C51)
        3. Use online net to select actions, target net to evaluate (Double)
        4. Network has Dueling architecture + Noisy layers (Dueling + Noisy)
        5. Update priorities based on TD-error (PER)
        """
        if len(self.memory) < max(BATCH_SIZE, MIN_MEMORY):
            return None, None, None

        # ── Step 1: Prioritized sampling ──
        self.per_beta = min(PER_BETA_END,
                           PER_BETA_START + (PER_BETA_END - PER_BETA_START)
                           * self.total_steps / PER_BETA_ANNEAL)
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(BATCH_SIZE, self.per_beta)

        # ── Step 2: Reset noise before forward pass ──
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # ── Step 3: Compute distributional target ──
        #
        # Standard DQN target:  r + γ * Q(s', argmax Q_online(s'))
        # Rainbow target:       project the shifted target distribution
        #
        # For each atom z_j in the target distribution:
        #   Tz_j = r + γ^n * z_j   (clipped to [V_MIN, V_MAX])
        # Then project Tz_j onto the fixed support grid.

        with torch.no_grad():
            # Double DQN: online net selects best action
            next_q = self.online_net.get_q_values(next_states, self.support)
            best_actions = next_q.argmax(1)                       # (batch,)

            # Target net evaluates that action's distribution
            next_log_probs = self.target_net(next_states)          # (batch, actions, atoms)
            next_probs = next_log_probs.exp()

            # Gather the distribution for the best action
            best_actions_expanded = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, NUM_ATOMS)
            next_dist = next_probs.gather(1, best_actions_expanded).squeeze(1)  # (batch, atoms)

            # ── C51 distributional projection ──
            # Shift support by reward and discount: Tz = r + γ^n * z
            gamma_n = GAMMA ** N_STEP
            Tz = rewards.unsqueeze(1) + gamma_n * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)
            Tz = Tz.clamp(V_MIN, V_MAX)                           # (batch, atoms)

            # Project onto fixed support
            b = (Tz - V_MIN) / self.delta_z                       # Fractional index
            l = b.floor().long()                                   # Lower bound
            u = b.ceil().long()                                    # Upper bound

            # Handle edge cases where l == u
            l = l.clamp(0, NUM_ATOMS - 1)
            u = u.clamp(0, NUM_ATOMS - 1)

            # Distribute probability to neighboring atoms
            target_dist = torch.zeros_like(next_dist)              # (batch, atoms)

            # Lower neighbor gets (u - b) fraction of the probability
            # Upper neighbor gets (b - l) fraction
            offset = torch.arange(BATCH_SIZE, device=device).unsqueeze(1).expand(-1, NUM_ATOMS)

            target_dist.view(-1).index_add_(
                0, (offset * NUM_ATOMS + l).view(-1),
                (next_dist * (u.float() - b)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (offset * NUM_ATOMS + u).view(-1),
                (next_dist * (b - l.float())).view(-1)
            )

        # ── Step 4: Compute loss ──
        log_probs = self.online_net(states)                        # (batch, actions, atoms)

        # Gather log-probs for the actions we took
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, NUM_ATOMS)
        chosen_log_probs = log_probs.gather(1, actions_expanded).squeeze(1)  # (batch, atoms)

        # Cross-entropy loss between predicted and target distributions
        # L = -Σ target_dist * log(predicted_dist)
        elementwise_loss = -(target_dist * chosen_log_probs).sum(1)    # (batch,)

        # ── Step 5: Apply importance-sampling weights (PER correction) ──
        loss = (weights * elementwise_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # ── Step 6: Update priorities ──
        td_errors = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Metrics
        with torch.no_grad():
            q_values = self.online_net.get_q_values(states, self.support)
            avg_q = q_values.gather(1, actions.unsqueeze(1)).mean().item()
        avg_priority = np.mean(td_errors)

        return loss.item(), avg_q, avg_priority

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition through n-step buffer, then into PER."""
        self.n_step_buffer.push(state, action, reward, next_state, done)

        if self.n_step_buffer.is_ready():
            s, a, R, ns, d = self.n_step_buffer.get()
            self.memory.push(s, a, R, ns, d)

        # Flush remaining at episode end
        if done:
            for transition in self.n_step_buffer.flush():
                self.memory.push(*transition)
            self.n_step_buffer.reset()

    def update_target_network(self):
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

    def get_noise_magnitude(self):
        """Measure current noise level (how much the network is exploring)."""
        total_sigma = 0.0
        count = 0
        for module in self.online_net.modules():
            if isinstance(module, NoisyLinear):
                total_sigma += module.weight_sigma.data.abs().mean().item()
                count += 1
        return total_sigma / max(count, 1)

    def record_episode(self, score, avg_loss, avg_q, avg_priority):
        self.score_history.append(score)
        self.loss_history.append(avg_loss)
        self.q_value_history.append(avg_q)
        self.priority_history.append(avg_priority)
        self.epsilon_history.append(self.get_noise_magnitude())

    def plot_training_curves(self, filename="rainbow_dqn_training_curves.png"):
        if len(self.score_history) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Rainbow DQN Training (Episode {self.episode})',
                    fontsize=14, fontweight='bold')

        episodes = range(1, len(self.score_history) + 1)

        def smooth(data, window=50):
            if len(data) < 10:
                return data, range(1, len(data) + 1)
            w = max(1, min(window, len(data) // 5))
            s = np.convolve(data, np.ones(w)/w, mode='valid')
            return s, range(w, len(data) + 1)

        # 1: Score
        ax = axes[0, 0]
        ax.plot(episodes, self.score_history, 'g-', alpha=0.2, linewidth=0.5)
        s, e = smooth(self.score_history)
        ax.plot(e, s, 'g-', linewidth=2, label='Smoothed')
        ax.axhline(y=self.best_score, color='r', linestyle='--', alpha=0.5,
                   label=f'Best: {int(self.best_score)}')
        ax.set_title('Episode Score')
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2: Loss (cross-entropy on distributions)
        ax = axes[0, 1]
        ax.plot(episodes, self.loss_history, 'b-', alpha=0.2, linewidth=0.5)
        s, e = smooth(self.loss_history)
        ax.plot(e, s, 'b-', linewidth=2, label='Smoothed')
        ax.set_title('Distribution Loss (C51 cross-entropy)')
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3: Average Q-value
        ax = axes[0, 2]
        ax.plot(episodes, self.q_value_history, 'orange', alpha=0.2, linewidth=0.5)
        s, e = smooth(self.q_value_history)
        ax.plot(e, s, 'orange', linewidth=2, label='Smoothed')
        ax.set_title('Average Q-value')
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4: PER average priority
        ax = axes[1, 0]
        ax.plot(episodes, self.priority_history, 'red', alpha=0.2, linewidth=0.5)
        s, e = smooth(self.priority_history)
        ax.plot(e, s, 'red', linewidth=2, label='Smoothed')
        ax.set_title('Avg Priority (PER)\n(decreasing = learning)')
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5: Noise magnitude (Noisy Net)
        ax = axes[1, 1]
        ax.plot(episodes, self.epsilon_history, 'purple', linewidth=2)
        ax.set_title('Noise σ (NoisyNet)\n(replaces ε-greedy)')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

        # 6: Component diagram
        ax = axes[1, 2]
        ax.axis('off')
        text = (
            "Rainbow DQN Components\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "1. Double DQN\n"
            "   online selects, target evaluates\n\n"
            "2. Dueling Architecture\n"
            "   Q = V + (A - mean A)\n\n"
            "3. Prioritized Replay\n"
            "   sample ∝ |TD-error|^α\n\n"
            "4. N-step Returns (n=3)\n"
            "   R = r₁+γr₂+γ²r₃+γ³V\n\n"
            "5. C51 Distributional\n"
            "   51 atoms in [-10, 10]\n\n"
            "6. Noisy Networks\n"
            "   learned exploration (no ε)"
        )
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
               fontsize=10, fontfamily='monospace',
               verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved to {filename}")

    def save(self, filename="rainbow_dqn_model.pth"):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'best_score': self.best_score,
            'total_steps': self.total_steps,
            'per_beta': self.per_beta,
            'score_history': self.score_history,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history,
            'priority_history': self.priority_history,
            'epsilon_history': self.epsilon_history,
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename="rainbow_dqn_model.pth"):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device, weights_only=False)
            self.online_net.load_state_dict(checkpoint['online_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.episode = checkpoint.get('episode', 0)
            self.best_score = checkpoint.get('best_score', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.per_beta = checkpoint.get('per_beta', PER_BETA_START)
            self.score_history = checkpoint.get('score_history', [])
            self.loss_history = checkpoint.get('loss_history', [])
            self.q_value_history = checkpoint.get('q_value_history', [])
            self.priority_history = checkpoint.get('priority_history', [])
            self.epsilon_history = checkpoint.get('epsilon_history', [])
            print(f"Model loaded from {filename}")
            return True
        return False


# ═══════════════════════════════════════════════════════════════
#  Game environment
# ═══════════════════════════════════════════════════════════════

class Bird:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 80
        self.y = SCREEN_HEIGHT // 2
        self.velocity_y = 0
        self.radius = 20

    def jump(self):
        self.velocity_y = JUMP_STRENGTH

    def update(self):
        self.velocity_y += GRAVITY
        self.y += self.velocity_y

    def draw(self, screen):
        pygame.draw.circle(screen, BIRD_COLOR, (int(self.x), int(self.y)), self.radius)
        eye_x = self.x + 8
        eye_y = self.y - 5
        pygame.draw.circle(screen, WHITE, (int(eye_x), int(eye_y)), 7)
        pygame.draw.circle(screen, BIRD_EYE, (int(eye_x + 2), int(eye_y)), 4)
        beak_points = [
            (self.x + self.radius - 5, self.y + 3),
            (self.x + self.radius + 12, self.y + 5),
            (self.x + self.radius - 5, self.y + 10)
        ]
        pygame.draw.polygon(screen, BIRD_BEAK, beak_points)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius + 5, self.y - self.radius + 5,
                          self.radius * 2 - 10, self.radius * 2 - 10)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(MIN_PIPE_HEIGHT + PIPE_GAP // 2,
                                     SCREEN_HEIGHT - GROUND_HEIGHT - MIN_PIPE_HEIGHT - PIPE_GAP // 2)
        self.passed = False

    def update(self, speed):
        self.x -= speed

    def draw(self, screen):
        top_height = self.gap_y - PIPE_GAP // 2
        pygame.draw.rect(screen, PIPE_COLOR, (self.x, 0, PIPE_WIDTH, top_height))
        pygame.draw.rect(screen, PIPE_HIGHLIGHT, (self.x, 0, 8, top_height))
        pygame.draw.rect(screen, PIPE_SHADOW, (self.x + PIPE_WIDTH - 8, 0, 8, top_height))
        pygame.draw.rect(screen, PIPE_COLOR, (self.x - 5, top_height - 30, PIPE_WIDTH + 10, 30))

        bottom_y = self.gap_y + PIPE_GAP // 2
        bottom_height = SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y
        pygame.draw.rect(screen, PIPE_COLOR, (self.x, bottom_y, PIPE_WIDTH, bottom_height))
        pygame.draw.rect(screen, PIPE_HIGHLIGHT, (self.x, bottom_y, 8, bottom_height))
        pygame.draw.rect(screen, PIPE_SHADOW, (self.x + PIPE_WIDTH - 8, bottom_y, 8, bottom_height))
        pygame.draw.rect(screen, PIPE_COLOR, (self.x - 5, bottom_y, PIPE_WIDTH + 10, 30))

    def get_rects(self):
        top_height = self.gap_y - PIPE_GAP // 2
        bottom_y = self.gap_y + PIPE_GAP // 2
        top_rect = pygame.Rect(self.x - 5, 0, PIPE_WIDTH + 10, top_height)
        bottom_rect = pygame.Rect(self.x - 5, bottom_y, PIPE_WIDTH + 10,
                                  SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y)
        return top_rect, bottom_rect

    def get_bottom_rect_pos(self):
        bottom_y = self.gap_y + PIPE_GAP // 2
        return self.x - 5, bottom_y

    def is_offscreen(self):
        return self.x + PIPE_WIDTH < 0


class Game:
    def __init__(self, render=True):
        self.render_game = render
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - Rainbow DQN")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 28)

        self.agent = RainbowAgent()
        self.training = True
        self.reset_game()

    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frame_count = 0
        self.game_over = False
        self.ground_offset = 0

        for i in range(4):
            self.pipes.append(Pipe(SCREEN_WIDTH + i * PIPE_SPACING))

    def get_state(self):
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                next_pipe = pipe
                break
        if next_pipe is None:
            next_pipe = self.pipes[-1]

        pipe_x, pipe_y = next_pipe.get_bottom_rect_pos()

        bird_y = (self.bird.y - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2)
        vel_y = self.bird.velocity_y / 15.0
        dx = (pipe_x - self.bird.x) / PIPE_SPACING
        dy = (self.bird.y - pipe_y) / (SCREEN_HEIGHT / 2)

        return torch.tensor([bird_y, vel_y, dx, dy], dtype=torch.float32)

    def draw_gradient_background(self):
        for y in range(SCREEN_HEIGHT - GROUND_HEIGHT):
            ratio = y / (SCREEN_HEIGHT - GROUND_HEIGHT)
            r = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * ratio)
            g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * ratio)
            b = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

    def draw_ground(self):
        pygame.draw.rect(self.screen, GROUND_TOP,
                        (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, 10))
        pygame.draw.rect(self.screen, GROUND_COLOR,
                        (0, SCREEN_HEIGHT - GROUND_HEIGHT + 10, SCREEN_WIDTH, GROUND_HEIGHT - 10))

    def draw_info(self):
        if not self.render_game:
            return

        info_texts = [
            ("Algorithm: Rainbow DQN", (255, 200, 50)),
            (f"Episode: {self.agent.episode}", WHITE),
            (f"Score: {int(self.score)}", WHITE),
            (f"Best: {int(self.agent.best_score)}", WHITE),
            (f"Memory: {len(self.agent.memory)}/{MEMORY_SIZE}", WHITE),
            (f"Steps: {self.agent.total_steps}", WHITE),
            (f"PER beta: {self.agent.per_beta:.3f}", (150, 200, 255)),
            (f"Noise sigma: {self.agent.get_noise_magnitude():.4f}", (200, 150, 255)),
            (f"Mode: {'Training' if self.training else 'Testing'}", WHITE),
        ]
        if self.agent.q_value_history:
            info_texts.append((f"Avg Q: {self.agent.q_value_history[-1]:.3f}", (255, 200, 100)))

        y = 10
        for text, color in info_texts:
            surface = self.font_small.render(text, True, color)
            shadow = self.font_small.render(text, True, BLACK)
            self.screen.blit(shadow, (12, y + 2))
            self.screen.blit(surface, (10, y))
            y += 25

        instructions = [
            "T: Toggle Training/Testing",
            "S: Save  L: Load  R: Reset",
            "ESC: Quit"
        ]
        y = SCREEN_HEIGHT - GROUND_HEIGHT - len(instructions) * 22 - 10
        for text in instructions:
            surface = self.font_small.render(text, True, (180, 180, 180))
            self.screen.blit(surface, (10, y))
            y += 22

    def check_collision(self):
        bird_rect = self.bird.get_rect()
        if self.bird.y + self.bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT:
            return True
        if self.bird.y - self.bird.radius < 0:
            return True
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.get_rects()
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                return True
        return False

    def step(self, action):
        if action == 1:
            self.bird.jump()

        self.bird.update()
        for pipe in self.pipes:
            pipe.update(BIRD_SPEED_X)

        passed_pipe = False
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
                passed_pipe = True

        if self.pipes and self.pipes[0].is_offscreen():
            self.pipes.pop(0)
            new_x = self.pipes[-1].x + PIPE_SPACING
            self.pipes.append(Pipe(new_x))

        done = self.check_collision()

        if done:
            reward = REWARD_DEAD
        else:
            reward = REWARD_ALIVE
            if passed_pipe:
                reward += REWARD_PASS_PIPE
            self.score += 1

        self.frame_count += 1
        return reward, done

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_t:
                    self.training = not self.training
                    print(f"Mode: {'Training' if self.training else 'Testing'}")
                if event.key == pygame.K_s:
                    self.agent.save()
                if event.key == pygame.K_l:
                    self.agent.load()
                if event.key == pygame.K_r:
                    self.agent = RainbowAgent()
                    print("Agent reset!")
        return True

    def run_episode(self):
        self.reset_game()
        state = self.get_state()
        episode_losses = []
        episode_q_values = []
        episode_priorities = []

        while True:
            if not self.handle_events():
                return None

            action = self.agent.get_action(state, training=self.training)
            reward, done = self.step(action)
            next_state = self.get_state()
            self.agent.total_steps += 1

            if self.training:
                # Store through n-step buffer → PER
                self.agent.store_transition(state, action, reward, next_state, done)

                if self.agent.total_steps % TRAIN_EVERY == 0:
                    result = self.agent.train_step()
                    if result[0] is not None:
                        loss, avg_q, avg_pri = result
                        episode_losses.append(loss)
                        episode_q_values.append(avg_q)
                        episode_priorities.append(avg_pri)

                self.agent.update_target_network()

            state = next_state

            if self.render_game:
                self.draw_gradient_background()
                for pipe in self.pipes:
                    pipe.draw(self.screen)
                self.draw_ground()
                self.bird.draw(self.screen)
                self.draw_info()
                pygame.display.flip()
                if self.training:
                    if TRAINING_FPS > 0:
                        self.clock.tick(TRAINING_FPS)
                else:
                    self.clock.tick(FPS)

            if done:
                break

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_q = np.mean(episode_q_values) if episode_q_values else 0
        avg_pri = np.mean(episode_priorities) if episode_priorities else 0
        return self.score, avg_loss, avg_q, avg_pri

    def run(self):
        print("=" * 58)
        print("  Flappy Bird — Rainbow DQN")
        print("  (Double + Dueling + PER + N-step + C51 + NoisyNet)")
        print("=" * 58)
        print(f"Device: {device}")
        print(f"C51: {NUM_ATOMS} atoms in [{V_MIN}, {V_MAX}]")
        print(f"N-step: {N_STEP}")
        print(f"PER: alpha={PER_ALPHA}, beta={PER_BETA_START}→{PER_BETA_END}")
        print(f"NoisyNet: sigma_init={NOISY_STD}")
        print("Controls:")
        print("  T - Toggle Training / Testing")
        print("  S - Save   L - Load   R - Reset")
        print("  ESC - Quit")
        print("=" * 58)

        self.agent.load()

        while True:
            if not self.handle_events():
                break

            self.agent.episode += 1
            result = self.run_episode()

            if result is None:
                break

            score, avg_loss, avg_q, avg_pri = result

            if self.training:
                self.agent.record_episode(score, avg_loss, avg_q, avg_pri)

            if score > self.agent.best_score:
                self.agent.best_score = score
                print(f"New best score: {int(score)} (Episode {self.agent.episode})")

            if self.agent.episode % 50 == 0:
                noise = self.agent.get_noise_magnitude()
                print(f"[Rainbow] Ep {self.agent.episode}: "
                      f"Score={int(score)}, Best={int(self.agent.best_score)}, "
                      f"Loss={avg_loss:.4f}, Q={avg_q:.3f}, "
                      f"Pri={avg_pri:.3f}, Noise={noise:.4f}, "
                      f"Beta={self.agent.per_beta:.3f}, "
                      f"Mem={len(self.agent.memory)}")
                if self.training:
                    self.agent.plot_training_curves()

            if self.agent.episode % 200 == 0:
                self.agent.save()

        self.agent.save()
        self.agent.plot_training_curves()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()
