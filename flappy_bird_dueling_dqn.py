"""
Flappy Bird with Dueling DQN (+ Double DQN)

Previous improvements changed the ALGORITHM:
  - Double DQN: fixed how target Q-values are computed
  - SARSA: changed the update rule to on-policy

Dueling DQN (Wang et al., 2016) changes the NETWORK ARCHITECTURE itself.

Key insight — decompose Q(s,a) into two streams:

    Q(s,a) = V(s) + A(s,a) - mean_a'(A(s,a'))

    V(s)   = State Value    → "how good is this state, regardless of action?"
    A(s,a) = Advantage      → "how much better is this action vs the average?"

Why this helps:
    In many states, the action choice barely matters (e.g., bird in open space
    between pipes). Standard DQN must learn Q(s,jump) ≈ Q(s,no_jump) ≈ high
    separately. Dueling learns V(s) = high once, shared across all actions.
    This makes learning more sample-efficient.

    The advantage stream only needs to learn the DIFFERENCE between actions,
    which is a simpler function — often near zero when actions don't matter.

Architecture:
    ┌─────────────┐
    │  Shared MLP  │
    │  (features)  │
    └──────┬───────┘
           │
     ┌─────┴─────┐
     │           │
  ┌──▼──┐   ┌──▼──────┐
  │  V   │   │  A      │
  │stream│   │ stream  │
  │→ V(s)│   │→ A(s,a) │
  └──┬───┘   └──┬──────┘
     │           │
     └─────┬─────┘
           │
    Q(s,a) = V(s) + A(s,a) - mean(A)

This implementation combines Dueling + Double DQN (the standard practice),
and supports toggling Dueling on/off for comparison.
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

# Hyperparameters
STATE_DIM = 4
HIDDEN_DIM = 128
BATCH_SIZE = 64
MEMORY_SIZE = 100000
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
TAU = 0.005
TRAIN_EVERY = 4
MIN_MEMORY = 1000

# Rewards
REWARD_ALIVE = 0.1
REWARD_DEAD = -1.0
REWARD_PASS_PIPE = 1.0

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════
#  Network Architectures — the core of this lesson
# ═══════════════════════════════════════════════════════════════

class StandardQNetwork(nn.Module):
    """
    Standard DQN: one stream, directly outputs Q(s,a).

        s → [fc1] → [fc2] → Q(s, a=0)
                           → Q(s, a=1)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN: two streams, computes Q = V + (A - mean(A)).

        s → [shared fc1] → [shared fc2] ─┬─→ [V stream]  → V(s)      (scalar)
                                          └─→ [A stream]  → A(s,a)    (per action)

        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    The "- mean(A)" is critical for IDENTIFIABILITY:
    Without it, V(s)+10 and A(s,a)-10 gives the same Q as V(s) and A(s,a).
    Subtracting the mean forces A to be centered at 0, giving V a clear meaning.
    """
    def __init__(self):
        super().__init__()

        # ─── Shared feature extraction ───
        self.shared_fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.shared_fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        # ─── Value stream: "how good is this state?" ───
        # Outputs a single scalar V(s)
        self.value_fc = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.value_out = nn.Linear(HIDDEN_DIM // 2, 1)

        # ─── Advantage stream: "how much better is each action?" ───
        # Outputs one value per action: A(s,a) for each a
        self.advantage_fc = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.advantage_out = nn.Linear(HIDDEN_DIM // 2, 2)

    def forward(self, x):
        # Shared feature extraction
        features = F.relu(self.shared_fc1(x))
        features = F.relu(self.shared_fc2(features))

        # Value stream
        v = F.relu(self.value_fc(features))
        v = self.value_out(v)                    # (batch, 1)

        # Advantage stream
        a = F.relu(self.advantage_fc(features))
        a = self.advantage_out(a)                # (batch, 2)

        # ★ Combine: Q = V + (A - mean(A))
        # Subtracting mean(A) ensures identifiability:
        #   - Forces advantages to be centered around 0
        #   - V(s) cleanly represents the state value
        #   - A(s,a) cleanly represents the relative advantage of each action
        q = v + (a - a.mean(dim=1, keepdim=True))

        return q

    def get_value_and_advantage(self, x):
        """Return V(s) and A(s,a) separately, for visualization."""
        features = F.relu(self.shared_fc1(x))
        features = F.relu(self.shared_fc2(features))

        v = F.relu(self.value_fc(features))
        v = self.value_out(v)

        a = F.relu(self.advantage_fc(features))
        a = self.advantage_out(a)

        return v, a


# ═══════════════════════════════════════════════════════════════
#  Replay Memory (same as Double DQN)
# ═══════════════════════════════════════════════════════════════

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )

    def __len__(self):
        return len(self.memory)


# ═══════════════════════════════════════════════════════════════
#  Agent — supports toggling Dueling on/off
# ═══════════════════════════════════════════════════════════════

class DuelingDQNAgent:
    def __init__(self, use_dueling=True):
        self.use_dueling = use_dueling

        # Create networks based on mode
        self._build_networks()

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON_START

        self.episode = 0
        self.best_score = 0
        self.total_steps = 0

        self.score_history = []
        self.loss_history = []
        self.q_value_history = []
        self.advantage_history = []    # ★ Track advantage magnitude (Dueling-specific)
        self.epsilon_history = []

    def _build_networks(self):
        """Build online and target networks based on current mode."""
        if self.use_dueling:
            self.online_net = DuelingQNetwork().to(device)
            self.target_net = DuelingQNetwork().to(device)
        else:
            self.online_net = StandardQNetwork().to(device)
            self.target_net = StandardQNetwork().to(device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)

    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_t = state.unsqueeze(0).to(device)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < max(BATCH_SIZE, MIN_MEMORY):
            return None, None, None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Current Q(s, a)
        q_values = self.online_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target (always use Double, it's strictly better)
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            targets = rewards + GAMMA * next_q * (1 - dones)

        loss = F.mse_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        avg_q = q_sa.mean().item()

        # ★ Track advantage magnitude (only meaningful for Dueling)
        avg_adv = 0.0
        if self.use_dueling and isinstance(self.online_net, DuelingQNetwork):
            with torch.no_grad():
                _, advantages = self.online_net.get_value_and_advantage(states)
                # How different are the actions? Large = action choice matters a lot
                avg_adv = (advantages.max(1)[0] - advantages.min(1)[0]).mean().item()

        return loss.item(), avg_q, avg_adv

    def update_target_network(self):
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def record_episode(self, score, avg_loss, avg_q, avg_adv):
        self.score_history.append(score)
        self.loss_history.append(avg_loss)
        self.q_value_history.append(avg_q)
        self.advantage_history.append(avg_adv)
        self.epsilon_history.append(self.epsilon)

    def plot_training_curves(self, filename="dueling_dqn_training_curves.png"):
        if len(self.score_history) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        mode = "Dueling + Double DQN" if self.use_dueling else "Standard + Double DQN"
        fig.suptitle(f'{mode} Training Progress (Episode {self.episode})', fontsize=14, fontweight='bold')

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

        # 2: Loss
        ax = axes[0, 1]
        ax.plot(episodes, self.loss_history, 'b-', alpha=0.2, linewidth=0.5)
        s, e = smooth(self.loss_history)
        ax.plot(e, s, 'b-', linewidth=2, label='Smoothed')
        ax.set_title('Training Loss')
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

        # 4: Advantage gap (★ Dueling-specific metric)
        ax = axes[1, 0]
        ax.plot(episodes, self.advantage_history, 'red', alpha=0.2, linewidth=0.5)
        s, e = smooth(self.advantage_history)
        ax.plot(e, s, 'red', linewidth=2, label='Smoothed')
        ax.set_title('Advantage Gap |A(best) - A(worst)|\n(small = action doesn\'t matter)')
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5: Epsilon
        ax = axes[1, 1]
        ax.plot(episodes, self.epsilon_history, 'purple', linewidth=2)
        ax.set_title('Exploration Rate (ε)')
        ax.set_xlabel('Episode')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # 6: Network architecture diagram (text)
        ax = axes[1, 2]
        ax.axis('off')
        if self.use_dueling:
            arch_text = (
                "Dueling Architecture\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "  state (4)\n"
                "     │\n"
                "  [shared 128]\n"
                "  [shared 128]\n"
                "     │\n"
                "  ┌──┴──┐\n"
                "  │     │\n"
                " [V]   [A]\n"
                " (64)  (64)\n"
                "  │     │\n"
                " V(s)  A(s,a)\n"
                " (1)   (2)\n"
                "  │     │\n"
                "  └──┬──┘\n"
                "     │\n"
                " Q = V + (A - mean(A))"
            )
        else:
            arch_text = (
                "Standard Architecture\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "  state (4)\n"
                "     │\n"
                "  [fc 128]\n"
                "  [fc 128]\n"
                "     │\n"
                "  [fc → 2]\n"
                "     │\n"
                "  Q(s,a)\n"
                "  (2 values)"
            )
        ax.text(0.5, 0.5, arch_text, transform=ax.transAxes,
               fontsize=11, fontfamily='monospace',
               verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved to {filename}")

    def save(self, filename="dueling_dqn_model.pth"):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode,
            'best_score': self.best_score,
            'total_steps': self.total_steps,
            'use_dueling': self.use_dueling,
            'score_history': self.score_history,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history,
            'advantage_history': self.advantage_history,
            'epsilon_history': self.epsilon_history,
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename="dueling_dqn_model.pth"):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device, weights_only=False)
            saved_dueling = checkpoint.get('use_dueling', True)
            # Rebuild networks if mode changed
            if saved_dueling != self.use_dueling:
                self.use_dueling = saved_dueling
                self._build_networks()
            self.online_net.load_state_dict(checkpoint['online_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', EPSILON_MIN)
            self.episode = checkpoint.get('episode', 0)
            self.best_score = checkpoint.get('best_score', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.score_history = checkpoint.get('score_history', [])
            self.loss_history = checkpoint.get('loss_history', [])
            self.q_value_history = checkpoint.get('q_value_history', [])
            self.advantage_history = checkpoint.get('advantage_history', [])
            self.epsilon_history = checkpoint.get('epsilon_history', [])
            print(f"Model loaded from {filename}")
            return True
        return False


# ═══════════════════════════════════════════════════════════════
#  Game environment (same as other implementations)
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
            pygame.display.set_caption("Flappy Bird - Dueling DQN")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 28)

        self.agent = DuelingDQNAgent(use_dueling=True)
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

        arch_str = "Dueling + Double DQN" if self.agent.use_dueling else "Standard + Double DQN"
        arch_color = (100, 255, 100) if self.agent.use_dueling else (255, 100, 100)

        info_texts = [
            (f"Architecture: {arch_str}", arch_color),
            (f"Episode: {self.agent.episode}", WHITE),
            (f"Score: {int(self.score)}", WHITE),
            (f"Best: {int(self.agent.best_score)}", WHITE),
            (f"Epsilon: {self.agent.epsilon:.4f}", WHITE),
            (f"Memory: {len(self.agent.memory)}/{MEMORY_SIZE}", WHITE),
            (f"Steps: {self.agent.total_steps}", WHITE),
            (f"Mode: {'Training' if self.training else 'Testing'}", WHITE),
        ]
        if self.agent.q_value_history:
            info_texts.append((f"Avg Q: {self.agent.q_value_history[-1]:.3f}", (255, 200, 100)))
        if self.agent.advantage_history and self.agent.use_dueling:
            info_texts.append((f"Adv Gap: {self.agent.advantage_history[-1]:.3f}", (255, 150, 150)))

        y = 10
        for text, color in info_texts:
            surface = self.font_small.render(text, True, color)
            shadow = self.font_small.render(text, True, BLACK)
            self.screen.blit(shadow, (12, y + 2))
            self.screen.blit(surface, (10, y))
            y += 25

        instructions = [
            "V: Toggle Dueling/Standard",
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
                if event.key == pygame.K_v:
                    # ★ Toggle architecture (resets networks!)
                    self.agent.use_dueling = not self.agent.use_dueling
                    self.agent._build_networks()
                    self.agent.epsilon = EPSILON_START
                    self.agent.score_history.clear()
                    self.agent.loss_history.clear()
                    self.agent.q_value_history.clear()
                    self.agent.advantage_history.clear()
                    self.agent.epsilon_history.clear()
                    mode = "Dueling" if self.agent.use_dueling else "Standard"
                    print(f"Switched to {mode} architecture (networks reset)")
                if event.key == pygame.K_t:
                    self.training = not self.training
                    print(f"Mode: {'Training' if self.training else 'Testing'}")
                if event.key == pygame.K_s:
                    self.agent.save()
                if event.key == pygame.K_l:
                    self.agent.load()
                if event.key == pygame.K_r:
                    dueling = self.agent.use_dueling
                    self.agent = DuelingDQNAgent(use_dueling=dueling)
                    print("Agent reset!")
        return True

    def run_episode(self):
        self.reset_game()
        state = self.get_state()
        episode_losses = []
        episode_q_values = []
        episode_adv_gaps = []

        while True:
            if not self.handle_events():
                return None

            action = self.agent.get_action(state, training=self.training)
            reward, done = self.step(action)
            next_state = self.get_state()
            self.agent.total_steps += 1

            if self.training:
                self.agent.memory.push(state, action, reward, next_state, done)

                if self.agent.total_steps % TRAIN_EVERY == 0:
                    result = self.agent.train_step()
                    if result[0] is not None:
                        loss, avg_q, avg_adv = result
                        episode_losses.append(loss)
                        episode_q_values.append(avg_q)
                        episode_adv_gaps.append(avg_adv)

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
        avg_adv = np.mean(episode_adv_gaps) if episode_adv_gaps else 0
        return self.score, avg_loss, avg_q, avg_adv

    def run(self):
        print("=" * 55)
        print("Flappy Bird — Dueling DQN vs Standard DQN")
        print("=" * 55)
        print(f"Device: {device}")
        print("Controls:")
        print("  V - Toggle Dueling / Standard architecture")
        print("  T - Toggle Training / Testing")
        print("  S - Save   L - Load   R - Reset")
        print("  ESC - Quit")
        print("=" * 55)

        self.agent.load()

        while True:
            if not self.handle_events():
                break

            self.agent.episode += 1
            result = self.run_episode()

            if result is None:
                break

            score, avg_loss, avg_q, avg_adv = result

            if self.training:
                self.agent.record_episode(score, avg_loss, avg_q, avg_adv)

            if score > self.agent.best_score:
                self.agent.best_score = score
                print(f"New best score: {int(score)} (Episode {self.agent.episode})")

            if self.training:
                self.agent.decay_epsilon()

            if self.agent.episode % 50 == 0:
                arch = "Dueling" if self.agent.use_dueling else "Standard"
                print(f"[{arch}] Ep {self.agent.episode}: "
                      f"Score={int(score)}, Best={int(self.agent.best_score)}, "
                      f"ε={self.agent.epsilon:.4f}, Loss={avg_loss:.4f}, "
                      f"Q={avg_q:.3f}, AdvGap={avg_adv:.3f}")
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
