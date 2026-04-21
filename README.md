# FlappyBirdRL

A comprehensive reinforcement learning tutorial built around Flappy Bird. Covers 13 algorithms from tabular Q-Learning to Rainbow DQN to GRPO, all implemented from scratch with PyTorch.

## Project Structure

```
flappy_bird.py              # Base game (human playable, press SPACE)
flappy_bird_qlearning.py    # Q-Learning (tabular)
flappy_bird_sarsa.py        # SARSA
flappy_bird_dqn.py          # DQN (CNN + frame stacking)
flappy_bird_double_dqn.py   # Double DQN
flappy_bird_dueling_dqn.py  # Dueling DQN
flappy_bird_rainbow_dqn.py  # Rainbow DQN
flappy_bird_pg.py           # REINFORCE (Policy Gradient)
flappy_bird_ac.py           # Actor-Critic
flappy_bird_ppo.py          # PPO
flappy_bird_a3c.py          # A3C
flappy_bird_grpo.py         # GRPO
flappy_bird_imitation.py    # Imitation Learning (BC + DAgger)
```

## Learning Roadmap

### Phase 1: Foundations

| # | Algorithm | File | Core Idea | Key Formula |
|---|-----------|------|-----------|-------------|
| 1 | **Q-Learning** | `flappy_bird_qlearning.py` | Store Q-value for every (state, action) pair in a table. Off-policy: assumes optimal future behavior. | `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]` |
| 2 | **DQN** | `flappy_bird_dqn.py` | Replace Q-table with a CNN. Experience replay + target network for stability. | Same as Q-Learning but Q is a neural network; uses frame stacking (4×84×84 grayscale) |
| 3 | **REINFORCE** | `flappy_bird_pg.py` | Directly optimize the policy. Sample trajectories, increase probability of actions that led to high returns. | `∇J = E[∇log π(a\|s) · R]` |
| 4 | **Actor-Critic** | `flappy_bird_ac.py` | Add a critic V(s) to reduce REINFORCE's variance. Actor learns policy, critic learns value. | `∇J = E[∇log π(a\|s) · (R - V(s))]` |
| 5 | **PPO** | `flappy_bird_ppo.py` | Clip the policy ratio to prevent destructively large updates. Stable and widely used. | `L = min(ratio·A, clip(ratio)·A)` with GAE advantages |
| 6 | **A3C** | `flappy_bird_a3c.py` | Multiple workers train in parallel with a shared global network. Asynchronous gradient updates. | Same as Actor-Critic but with N parallel workers and async updates |

### Phase 2: Value-Based Improvements

| # | Algorithm | File | What It Fixes | Key Change |
|---|-----------|------|---------------|------------|
| 7 | **SARSA** | `flappy_bird_sarsa.py` | On-policy alternative to Q-Learning. Uses the action actually taken, not the theoretical best. More conservative. | `Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]` where a' is the actual next action |
| 8 | **Double DQN** | `flappy_bird_double_dqn.py` | Q-value overestimation. Same network selecting and evaluating → biased high. | Online net selects action, target net evaluates: `a* = argmax Q_online(s'); target = Q_target(s', a*)` |
| 9 | **Dueling DQN** | `flappy_bird_dueling_dqn.py` | Inefficient value learning. In many states, action choice barely matters — but standard DQN must learn Q for each action separately. | Split network: `Q(s,a) = V(s) + A(s,a) - mean(A)`. V stream learns state value, A stream learns action advantage. |
| 10 | **Rainbow DQN** | `flappy_bird_rainbow_dqn.py` | Combines 6 improvements into one agent. | Double + Dueling + PER + N-step + C51 + NoisyNet |

#### Rainbow DQN Components

| Component | Problem It Solves | Mechanism |
|-----------|-------------------|-----------|
| Double DQN | Q-value overestimation | Decouple action selection (online net) from evaluation (target net) |
| Dueling DQN | Inefficient state/action separation | Two-stream architecture: V(s) + A(s,a) |
| Prioritized Experience Replay | Uniform sampling wastes data | Sample proportional to TD-error; importance-sampling correction |
| N-step Returns (n=3) | Slow reward propagation | `R = r₁ + γr₂ + γ²r₃ + γ³V(s₄)` instead of `R = r + γV(s')` |
| C51 (Distributional) | Learning only expected value ignores risk | Learn full return distribution over 51 atoms in [-10, 10] |
| Noisy Networks | Crude ε-greedy exploration | Learnable noise in weights: `w = μ + σ×ε`. Network learns when to explore. |

### Phase 3: Policy Alignment & Imitation

| # | Algorithm | File | Core Idea |
|---|-----------|------|-----------|
| 11 | **GRPO** | `flappy_bird_grpo.py` | Remove the critic entirely. Sample G trajectories, use group-relative normalization as advantage: `A_i = (score_i - mean) / std`. Natural fit for environments with scalar episode rewards. |
| 12 | **BC** | `flappy_bird_imitation.py` | Behavioral Cloning: supervised learning on expert demonstrations. Simple but suffers from distribution shift. |
| 13 | **DAgger** | `flappy_bird_imitation.py` | Fixes BC's distribution shift by iteratively: learner plays → expert labels → aggregate dataset → retrain. |

#### LLM Alignment Algorithms (Theory Only)

These algorithms are designed for LLM alignment and don't naturally fit game environments (they need human preference data, not scalar rewards). Covered as theory during the learning process:

| Algorithm | Core Idea |
|-----------|-----------|
| **RLHF** | Three-stage pipeline: SFT → train Reward Model from preference pairs → PPO with RM reward + KL penalty. `reward = RM(s) - β·KL(π \|\| π_ref)` |
| **DPO** | Collapses RLHF into one step, no reward model needed: `L = -log σ(β·[log π(win)/π_ref(win) - log π(lose)/π_ref(lose)])` |

## Algorithm Taxonomy

```
                          RL Algorithms
                               │
              ┌────────────────┼────────────────┐
         Value-Based      Policy-Based      Imitation
              │                │                │
   ┌──────┬──┴──┐     ┌──────┼──────┐     ┌───┴───┐
Q-Learn SARSA  DQN    PG    AC    PPO     BC   DAgger
               │             │      │
        ┌──────┼──────┐     A3C   GRPO
      Double Dueling Rainbow

                    LLM Alignment (theory only)
                          ┌────┴────┐
                        RLHF       DPO
```

## Key Concepts

### Value-Based vs Policy-Based

| | Value-Based | Policy-Based |
|---|---|---|
| Learns | Q(s,a) — how much each action is worth | π(a\|s) — what action to take |
| Action selection | argmax Q (deterministic) | Sample from probability (stochastic) |
| Exploration | Needs external ε-greedy | Built-in (probabilistic output) |
| Action space | Discrete only | Discrete and continuous |
| In this project | Q-Learning, SARSA, DQN, Double/Dueling/Rainbow | REINFORCE |
| Hybrid (Actor-Critic) | Critic provides V(s) | Actor provides π(a\|s) |
| Hybrid algorithms | | AC, PPO, A3C, GRPO |

### On-Policy vs Off-Policy

| | On-Policy | Off-Policy |
|---|---|---|
| Data source | Must come from current policy | Any source (old data, other policies) |
| Data reuse | Use a few times, then discard | Store in Replay Buffer, reuse many times |
| Data efficiency | Low (lots of data used briefly) | High (same data reused extensively) |
| Stability | High (data matches policy) | Needs extra tricks (target network, etc.) |
| In this project | SARSA, REINFORCE, AC, PPO, A3C, GRPO | Q-Learning, DQN, Double/Dueling/Rainbow |

## State Representations

| Approach | Used By | Dimensions | Description |
|----------|---------|-----------|-------------|
| Discrete bins | Q-Learning, SARSA | 20×20×10 = 4000 states | Discretized (dx, dy, velocity) |
| 4-dim vector | PG, AC, PPO, A3C, Double/Dueling/Rainbow DQN, GRPO, Imitation | 4 floats | `[bird_y, velocity_y, pipe_dx, pipe_dy]` normalized to ~[-1,1] |
| Frame stacking | DQN (CNN) | 4×84×84 | 4 consecutive grayscale frames |

## Quick Start

```bash
# Play the game yourself
python flappy_bird.py

# Train with a specific algorithm (press T to start training)
python flappy_bird_qlearning.py
python flappy_bird_ppo.py
python flappy_bird_rainbow_dqn.py
python flappy_bird_grpo.py

# Imitation learning (trains expert first, then BC + DAgger)
python flappy_bird_imitation.py
```

### Common Controls

| Key | Action |
|-----|--------|
| `T` | Toggle Training / Testing |
| `S` | Save model |
| `L` | Load model |
| `R` | Reset agent |
| `V` | Toggle algorithm variant (Double/Dueling files) |
| `ESC` | Quit |

## Dependencies

- Python 3.8+
- PyTorch
- Pygame
- NumPy
- Matplotlib
