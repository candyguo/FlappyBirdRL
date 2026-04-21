"""
Flappy Bird with Imitation Learning (BC + DAgger)

Imitation Learning: learn by watching an expert, not by trial-and-error.

Two algorithms implemented here:

1. BC (Behavioral Cloning):
     - Record expert's (state, action) pairs
     - Train policy as supervised classification: state → action
     - Problem: distribution shift (compounding errors)

2. DAgger (Dataset Aggregation, Ross et al., 2011):
     - Run LEARNER's policy to collect states
     - Ask EXPERT to label those states
     - Aggregate into growing dataset → retrain
     - Fixes BC's distribution shift because the learner trains on
       states it actually encounters, including mistakes

   BC:     train on expert's states only     → fragile
   DAgger: train on learner's states too     → robust

Real-world analogies:
   BC     = reading a textbook (only see ideal examples)
   DAgger = having a driving instructor in the passenger seat
            (you drive, they correct you in real-time)

Expert source (choose one):
  - Train a PPO expert from scratch
  - Load a pre-trained Rainbow DQN model (or any other saved model)
Press 1/2/3 to toggle between Expert, BC, and DAgger.
"""

import pygame
import random
import sys
import os
import numpy as np
from collections import deque
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
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

# ─── Hyperparameters ─────────────────────────────────────────
STATE_DIM = 4
HIDDEN_DIM = 128

# Expert training (PPO)
EXPERT_EPISODES = 500
EXPERT_LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64
ROLLOUT_STEPS = 2048

# Imitation learning
IL_LEARNING_RATE = 1e-3
BC_EXPERT_EPISODES = 200        # How many expert episodes to record for BC
DAGGER_ROUNDS = 20              # Number of DAgger iterations
DAGGER_EPISODES_PER_ROUND = 50  # Episodes per DAgger round
TRAJECTORY_MAX_STEPS = 500      # Max steps per episode (prevents hanging with strong experts)
IL_TRAIN_EPOCHS = 10            # Training epochs per round
IL_BATCH_SIZE = 128

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════
#  Rainbow DQN Expert Wrapper
# ═══════════════════════════════════════════════════════════════
#
#  Loads a pre-trained Rainbow DQN model and wraps it to provide
#  the same interface as the PPO expert: get_expert_action(state) → action

def load_rainbow_expert(model_path="rainbow_dqn_model.pth"):
    """
    Load a trained Rainbow DQN and wrap it as an expert.
    Returns None if the model file doesn't exist.
    """
    if not os.path.exists(model_path):
        return None

    # Import Rainbow components (same file must be in the same directory)
    from flappy_bird_rainbow_dqn import RainbowNetwork, NUM_ATOMS, V_MIN, V_MAX

    class RainbowExpertWrapper:
        """
        Wraps Rainbow DQN to match expert interface.

        Rainbow outputs: probability distribution over 51 atoms per action
        Expert needs:    a single best action

        Conversion: Q(s,a) = Σ pᵢ × zᵢ → argmax over actions
        """
        def __init__(self, network, support):
            self.network = network
            self.support = support
            self.network.eval()

        def get_expert_action(self, state):
            with torch.no_grad():
                q_values = self.network.get_q_values(state, self.support)
                return q_values.argmax(dim=-1).item()

    # Load model
    network = RainbowNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint['online_net'])
    network.eval()

    support = torch.linspace(V_MIN, V_MAX, NUM_ATOMS).to(device)
    expert = RainbowExpertWrapper(network, support)

    best_score = checkpoint.get('best_score', 0)
    episode = checkpoint.get('episode', 0)
    print(f"  Rainbow expert loaded: {model_path} "
          f"(episode={episode}, best={int(best_score)})")
    return expert


# ═══════════════════════════════════════════════════════════════
#  Policy Network
# ═══════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """Simple policy network: state → action probabilities."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(HIDDEN_DIM, 2)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)  # Only used for PPO expert

    def forward(self, x):
        features = self.net(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action(self, state, greedy=False):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def get_expert_action(self, state):
        """Expert always acts greedily (no exploration)."""
        with torch.no_grad():
            logits, _ = self.forward(state)
            return logits.argmax(dim=-1).item()

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy()


# ═══════════════════════════════════════════════════════════════
#  Learner Network (for BC / DAgger — no value head needed)
# ═══════════════════════════════════════════════════════════════

class LearnerNetwork(nn.Module):
    """
    Learner policy for imitation learning.
    Same as PolicyNetwork but without value_head (no RL needed).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2),
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, state, greedy=False):
        logits = self.forward(state)
        if greedy:
            return logits.argmax(dim=-1).item()
        dist = Categorical(logits=logits)
        return dist.sample().item()


# ═══════════════════════════════════════════════════════════════
#  Game Environment
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
        return (pygame.Rect(self.x - 5, 0, PIPE_WIDTH + 10, top_height),
                pygame.Rect(self.x - 5, bottom_y, PIPE_WIDTH + 10,
                            SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y))

    def get_bottom_rect_pos(self):
        return self.x - 5, self.gap_y + PIPE_GAP // 2

    def is_offscreen(self):
        return self.x + PIPE_WIDTH < 0


class GameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        for i in range(4):
            self.pipes.append(Pipe(SCREEN_WIDTH + i * PIPE_SPACING))
        return self.get_state()

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

    def step(self, action):
        if action == 1:
            self.bird.jump()
        self.bird.update()
        for pipe in self.pipes:
            pipe.update(BIRD_SPEED_X)
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
        if self.pipes and self.pipes[0].is_offscreen():
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1].x + PIPE_SPACING))
        done = self._check_collision()
        if not done:
            self.score += 1
        return self.get_state(), done

    def _check_collision(self):
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


# ═══════════════════════════════════════════════════════════════
#  Step 0: Train Expert (PPO)
# ═══════════════════════════════════════════════════════════════

def train_expert(num_episodes=EXPERT_EPISODES):
    """Train an expert policy with PPO. This simulates having a human expert."""
    print("=" * 55)
    print("  Step 0: Training Expert Policy (PPO)")
    print("=" * 55)

    expert = PolicyNetwork().to(device)
    optimizer = optim.Adam(expert.parameters(), lr=EXPERT_LR)
    env = GameEnv()
    best_score = 0

    for ep in range(1, num_episodes + 1):
        states, actions, log_probs_old, values, rewards, dones = \
            [], [], [], [], [], []
        state = env.reset()

        for _ in range(ROLLOUT_STEPS):
            state_t = state.unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = expert.get_action(state_t)
            next_state, done = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs_old.append(log_prob)
            values.append(value)
            rewards.append(0.1 if not done else -1.0)
            dones.append(done)

            state = next_state
            if done:
                if env.score > best_score:
                    best_score = env.score
                state = env.reset()

        # PPO update
        states_t = torch.stack(states).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(device)
        old_lp = torch.cat(log_probs_old).detach()
        vals = torch.cat(values).detach()
        rews = torch.tensor(rewards, dtype=torch.float32).to(device)
        dns = torch.tensor(dones, dtype=torch.float32).to(device)

        # GAE — split rollout into episodes, compute each independently
        advantages = torch.zeros_like(rews)
        returns = torch.zeros_like(rews)

        # Step 1: Find episode boundaries
        #   dones = [F, F, T, F, F, F, T, F, T, ...]
        #   episodes = [(0,2), (3,6), (7,8), ...]  each is (start, end) inclusive
        episodes = []
        ep_start = 0
        for t in range(len(dns)):
            if dns[t] == 1 or t == len(dns) - 1:
                episodes.append((ep_start, t))
                ep_start = t + 1

        # Step 2: Compute GAE for each episode separately
        for (start, end) in episodes:
            ep_rews = rews[start:end + 1]
            ep_vals = vals[start:end + 1]
            ep_len = end - start + 1

            # Last step's next value:
            #   if episode ended (done=True): next_V = 0 (no future)
            #   if rollout truncated (last step, done=False): next_V = V(s_last) from network
            if dns[end] == 1:
                next_v = 0.0
            else:
                with torch.no_grad():
                    _, nv = expert(states_t[end:end + 1])
                    next_v = nv.squeeze().item()

            # Compute GAE backwards within this episode only
            gae = 0.0
            for t in reversed(range(ep_len)):
                if t == ep_len - 1:
                    next_val = next_v
                else:
                    next_val = ep_vals[t + 1]

                delta = ep_rews[t] + GAMMA * next_val - ep_vals[t]
                gae = delta + GAMMA * LAMBDA * gae
                advantages[start + t] = gae

        returns = advantages + vals

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(len(states_t))
            for start in range(0, len(states_t), MINI_BATCH_SIZE):
                mb = idx[start:start + MINI_BATCH_SIZE]
                new_lp, new_v, ent = expert.evaluate(states_t[mb], actions_t[mb])
                ratio = (new_lp - old_lp[mb]).exp()
                adv = (advantages[mb] - advantages[mb].mean()) / (advantages[mb].std() + 1e-8)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(new_v, returns[mb]) - 0.01 * ent.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(expert.parameters(), 0.5)
                optimizer.step()

        if ep % 50 == 0:
            print(f"  Expert Episode {ep}/{num_episodes}: Best={int(best_score)}")

    print(f"  Expert trained. Best score: {int(best_score)}")
    return expert


# ═══════════════════════════════════════════════════════════════
#  Step 1: Behavioral Cloning (BC)
# ═══════════════════════════════════════════════════════════════
#
#  The simplest imitation learning:
#    1. Expert plays → record (state, action) dataset
#    2. Supervised learning: minimize CrossEntropy(π(s), a_expert)
#
#  That's it. No RL, no reward, no environment interaction during training.

def collect_expert_demonstrations(expert, num_episodes=BC_EXPERT_EPISODES):
    """
    Record expert playing the game.

    In real world: a human plays and we record their inputs.
    Here: PPO expert plays and we record (state, action) pairs.
    """
    print(f"  Recording {num_episodes} expert demonstrations...")
    env = GameEnv()
    all_states = []
    all_actions = []
    total_score = 0

    for i in range(num_episodes):
        state = env.reset()
        steps = 0
        while steps < TRAJECTORY_MAX_STEPS:
            state_t = state.unsqueeze(0).to(device)
            action = expert.get_expert_action(state_t)

            all_states.append(state)
            all_actions.append(action)

            state, done = env.step(action)
            steps += 1
            if done:
                break
        total_score += env.score

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{num_episodes} episodes recorded")

    avg_score = total_score / num_episodes
    print(f"  Recorded {len(all_states)} state-action pairs "
          f"(avg expert score: {avg_score:.0f})")

    return torch.stack(all_states), torch.tensor(all_actions, dtype=torch.long)


def train_bc(expert_states, expert_actions, num_epochs=IL_TRAIN_EPOCHS):
    """
    Behavioral Cloning: pure supervised learning.

    loss = CrossEntropy(learner_logits, expert_action)

    Same as training an image classifier:
        image → CNN → class label
        state → MLP → action label
    """
    print("\n" + "=" * 55)
    print("  Training BC (Behavioral Cloning)")
    print("=" * 55)

    learner = LearnerNetwork().to(device)
    optimizer = optim.Adam(learner.parameters(), lr=IL_LEARNING_RATE)

    states = expert_states.to(device)
    actions = expert_actions.to(device)
    dataset_size = len(states)

    for epoch in range(1, num_epochs + 1):
        indices = torch.randperm(dataset_size)
        total_loss = 0
        correct = 0

        for start in range(0, dataset_size, IL_BATCH_SIZE):
            idx = indices[start:start + IL_BATCH_SIZE]
            logits = learner(states[idx])

            # Cross-entropy loss: treat expert action as ground truth label
            loss = F.cross_entropy(logits, actions[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == actions[idx]).sum().item()

        acc = correct / dataset_size
        avg_loss = total_loss / dataset_size
        print(f"  BC Epoch {epoch}/{num_epochs}: loss={avg_loss:.4f}, accuracy={acc:.2%}")

    return learner


# ═══════════════════════════════════════════════════════════════
#  Step 2: DAgger (Dataset Aggregation)
# ═══════════════════════════════════════════════════════════════
#
#  Fixes BC's distribution shift by iteratively:
#    1. Learner plays (visits its OWN states, including mistakes)
#    2. Expert labels those states ("what would you do here?")
#    3. Add to dataset → retrain
#
#  Over time, the dataset covers states the learner actually visits,
#  including recovery from mistakes.

def train_dagger(expert, num_rounds=DAGGER_ROUNDS):
    """
    DAgger: iterative imitation learning.

    Round 1: Start with BC dataset + train
    Round 2: Learner plays → expert labels → aggregate → retrain
    Round 3: Same (dataset keeps growing)
    ...

    Key insight: each round, the learner visits new states
    (including states it reaches by making mistakes).
    The expert labels these → learner learns to recover.
    """
    print("\n" + "=" * 55)
    print("  Training DAgger (Dataset Aggregation)")
    print("=" * 55)

    env = GameEnv()
    learner = LearnerNetwork().to(device)
    optimizer = optim.Adam(learner.parameters(), lr=IL_LEARNING_RATE)

    # Start with some expert demonstrations (same as BC)
    all_states, all_actions = collect_expert_demonstrations(
        expert, num_episodes=BC_EXPERT_EPISODES // 2)
    all_states = all_states.to(device)
    all_actions = all_actions.to(device)

    scores_per_round = []

    for round_num in range(1, num_rounds + 1):
        # ── Phase 1: Learner plays, expert labels ──
        new_states = []
        new_actions = []
        round_scores = []

        for _ in range(DAGGER_EPISODES_PER_ROUND):
            state = env.reset()
            steps = 0
            while steps < TRAJECTORY_MAX_STEPS:
                state_t = state.unsqueeze(0).to(device)

                # ★ LEARNER chooses action (visits its own distribution)
                with torch.no_grad():
                    learner_action = learner.get_action(state_t, greedy=True)

                # ★ EXPERT labels this state (what would expert do here?)
                expert_action = expert.get_expert_action(state_t)

                new_states.append(state)
                new_actions.append(expert_action)  # Store EXPERT's action, not learner's

                # Execute LEARNER's action (to stay in learner's distribution)
                state, done = env.step(learner_action)
                steps += 1
                if done:
                    break
            round_scores.append(env.score)

        # ── Phase 2: Aggregate dataset ──
        new_s = torch.stack(new_states).to(device)
        new_a = torch.tensor(new_actions, dtype=torch.long).to(device)

        # ★ Dataset grows each round (this is the "Aggregation" in DAgger)
        all_states = torch.cat([all_states, new_s])
        all_actions = torch.cat([all_actions, new_a])

        # ── Phase 3: Retrain on full dataset ──
        dataset_size = len(all_states)
        for epoch in range(IL_TRAIN_EPOCHS):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, IL_BATCH_SIZE):
                idx = indices[start:start + IL_BATCH_SIZE]
                logits = learner(all_states[idx])
                loss = F.cross_entropy(logits, all_actions[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_score = np.mean(round_scores)
        scores_per_round.append(avg_score)
        print(f"  DAgger Round {round_num}/{num_rounds}: "
              f"AvgScore={avg_score:.0f}, Dataset={dataset_size}, "
              f"BestRound={max(round_scores)}")

    return learner, scores_per_round


# ═══════════════════════════════════════════════════════════════
#  Evaluation: Compare BC vs DAgger
# ═══════════════════════════════════════════════════════════════

def evaluate_policy(policy, env, num_episodes=100, name="Policy"):
    """Evaluate a policy over many episodes."""
    scores = []
    for _ in range(num_episodes):
        state = env.reset()
        steps = 0
        while steps < TRAJECTORY_MAX_STEPS:
            state_t = state.unsqueeze(0).to(device)
            with torch.no_grad():
                if isinstance(policy, LearnerNetwork):
                    action = policy.get_action(state_t, greedy=True)
                else:
                    action = policy.get_expert_action(state_t)
            state, done = env.step(action)
            steps += 1
            if done:
                break
        scores.append(env.score)
    avg = np.mean(scores)
    std = np.std(scores)
    print(f"  {name}: avg={avg:.0f}, std={std:.0f}, "
          f"min={min(scores)}, max={max(scores)}")
    return scores


def plot_comparison(expert_scores, bc_scores, dagger_scores, dagger_rounds,
                    filename="imitation_learning_comparison.png"):
    """Plot BC vs DAgger comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Imitation Learning: BC vs DAgger', fontsize=14, fontweight='bold')

    # 1: Score distribution comparison
    ax = axes[0]
    ax.hist(expert_scores, bins=20, alpha=0.5, label=f'Expert (avg={np.mean(expert_scores):.0f})', color='green')
    ax.hist(bc_scores, bins=20, alpha=0.5, label=f'BC (avg={np.mean(bc_scores):.0f})', color='blue')
    ax.hist(dagger_scores, bins=20, alpha=0.5, label=f'DAgger (avg={np.mean(dagger_scores):.0f})', color='red')
    ax.set_title('Score Distribution')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2: DAgger learning curve (score per round)
    ax = axes[1]
    ax.plot(range(1, len(dagger_rounds) + 1), dagger_rounds, 'r-o', linewidth=2)
    ax.axhline(y=np.mean(bc_scores), color='blue', linestyle='--',
              label=f'BC avg={np.mean(bc_scores):.0f}')
    ax.axhline(y=np.mean(expert_scores), color='green', linestyle='--',
              label=f'Expert avg={np.mean(expert_scores):.0f}')
    ax.set_title('DAgger Learning Curve')
    ax.set_xlabel('Round')
    ax.set_ylabel('Avg Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3: Algorithm comparison text
    ax = axes[2]
    ax.axis('off')
    text = (
        "Imitation Learning\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "BC (Behavioral Cloning)\n"
        "  Train on: expert states\n"
        "  Loss: CrossEntropy\n"
        "  Problem: distribution shift\n\n"
        "DAgger (Dataset Aggregation)\n"
        "  Train on: learner states\n"
        "  + expert labels\n"
        "  Fix: covers mistake states\n\n"
        f"Expert avg: {np.mean(expert_scores):.0f}\n"
        f"BC avg:     {np.mean(bc_scores):.0f}\n"
        f"DAgger avg: {np.mean(dagger_scores):.0f}"
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
           fontsize=12, fontfamily='monospace',
           verticalalignment='center', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved to {filename}")


# ═══════════════════════════════════════════════════════════════
#  Visual Game
# ═══════════════════════════════════════════════════════════════

class Game:
    def __init__(self, render=True):
        self.render_game = render
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - Imitation Learning")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 28)

        self.bc_policy = None
        self.dagger_policy = None
        self.expert_policy = None
        self.active_policy = None
        self.active_name = "None"
        self.reset_game()

    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
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

        color_map = {
            "Expert": (100, 255, 100),
            "BC": (100, 150, 255),
            "DAgger": (255, 150, 100),
        }
        name_color = color_map.get(self.active_name, WHITE)

        info_texts = [
            (f"Playing: {self.active_name}", name_color),
            (f"Score: {int(self.score)}", WHITE),
        ]

        y = 10
        for text, color in info_texts:
            surface = self.font_small.render(text, True, color)
            shadow = self.font_small.render(text, True, BLACK)
            self.screen.blit(shadow, (12, y + 2))
            self.screen.blit(surface, (10, y))
            y += 25

        instructions = [
            "1: Watch Expert",
            "2: Watch BC",
            "3: Watch DAgger",
            "ESC: Quit"
        ]
        y = SCREEN_HEIGHT - GROUND_HEIGHT - len(instructions) * 22 - 10
        for text in instructions:
            surface = self.font_small.render(text, True, (180, 180, 180))
            self.screen.blit(surface, (10, y))
            y += 22

    def step_game(self, action):
        if action == 1:
            self.bird.jump()
        self.bird.update()
        for pipe in self.pipes:
            pipe.update(BIRD_SPEED_X)
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
        if self.pipes and self.pipes[0].is_offscreen():
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1].x + PIPE_SPACING))
        done = False
        bird_rect = self.bird.get_rect()
        if self.bird.y + self.bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT:
            done = True
        if self.bird.y - self.bird.radius < 0:
            done = True
        for pipe in self.pipes:
            tr, br = pipe.get_rects()
            if bird_rect.colliderect(tr) or bird_rect.colliderect(br):
                done = True
        if not done:
            self.score += 1
        return done

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_1 and self.expert_policy:
                    self.active_policy = self.expert_policy
                    self.active_name = "Expert"
                    print("Switched to Expert")
                if event.key == pygame.K_2 and self.bc_policy:
                    self.active_policy = self.bc_policy
                    self.active_name = "BC"
                    print("Switched to BC")
                if event.key == pygame.K_3 and self.dagger_policy:
                    self.active_policy = self.dagger_policy
                    self.active_name = "DAgger"
                    print("Switched to DAgger")
        return True

    def run_visual_episode(self):
        self.reset_game()
        state = self.get_state()

        while True:
            if not self.handle_events():
                return None

            if self.active_policy is None:
                return 0

            state_t = state.unsqueeze(0).to(device)
            with torch.no_grad():
                if isinstance(self.active_policy, LearnerNetwork):
                    action = self.active_policy.get_action(state_t, greedy=True)
                else:
                    action = self.active_policy.get_expert_action(state_t)

            done = self.step_game(action)
            state = self.get_state()

            if self.render_game:
                self.draw_gradient_background()
                for pipe in self.pipes:
                    pipe.draw(self.screen)
                self.draw_ground()
                self.bird.draw(self.screen)
                self.draw_info()
                pygame.display.flip()
                self.clock.tick(FPS)

            if done:
                break
        return self.score

    def run(self):
        print("\n" + "=" * 58)
        print("  Flappy Bird — Imitation Learning (BC + DAgger)")
        print("=" * 58)

        # ── Step 0: Get expert ──
        # Try loading Rainbow DQN first, fall back to training PPO
        self.expert_policy = load_rainbow_expert("rainbow_dqn_model.pth")
        if self.expert_policy is not None:
            print("  Using Rainbow DQN as expert")
        else:
            print("  No Rainbow model found, training PPO expert...")
            self.expert_policy = train_expert()

        # ── Step 1: BC ──
        expert_states, expert_actions = collect_expert_demonstrations(self.expert_policy)
        self.bc_policy = train_bc(expert_states, expert_actions)

        # ── Step 2: DAgger ──
        self.dagger_policy, dagger_round_scores = train_dagger(self.expert_policy)

        # ── Evaluate all three ──
        print("\n" + "=" * 55)
        print("  Evaluation (100 episodes each)")
        print("=" * 55)
        env = GameEnv()
        expert_scores = evaluate_policy(self.expert_policy, env, name="Expert")
        bc_scores = evaluate_policy(self.bc_policy, env, name="BC")
        dagger_scores = evaluate_policy(self.dagger_policy, env, name="DAgger")

        plot_comparison(expert_scores, bc_scores, dagger_scores, dagger_round_scores)

        # ── Visual comparison ──
        print("\n" + "=" * 55)
        print("  Visual Mode — Press 1/2/3 to switch policies")
        print("  1: Expert  2: BC  3: DAgger")
        print("=" * 55)

        self.active_policy = self.expert_policy
        self.active_name = "Expert"

        while True:
            score = self.run_visual_episode()
            if score is None:
                break

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()
