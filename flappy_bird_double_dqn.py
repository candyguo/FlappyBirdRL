"""
Flappy Bird with Double DQN

Solves the Q-value OVERESTIMATION problem in vanilla DQN.

Problem in Vanilla DQN:
    target = r + γ * max_a Q_target(s', a)
    The max operator uses the SAME network to both SELECT and EVALUATE actions.
    Noisy Q-estimates cause max to consistently pick overestimated values.
    → Q-values inflate → policy degrades.

Double DQN fix (van Hasselt et al., 2016):
    a*     = argmax_a Q_online(s', a)       ← online net SELECTS best action
    target = r + γ * Q_target(s', a*)       ← target net EVALUATES that action
    Two networks cross-check each other, suppressing overestimation.

This file trains TWO independent agents in parallel on the same game:
  - Agent A: Vanilla DQN
  - Agent B: Double DQN
Both see identical states but learn independently.
A comparison plot is saved showing Q-value overestimation differences.

Uses 4-dim state vector [bird_y, velocity_y, pipe_dx, pipe_dy] (same as PG/AC/PPO)
for simplicity — focus is on the algorithm, not the network architecture.
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
TRAINING_FPS = 0  # No limit during training

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

# DQN Hyperparameters
STATE_DIM = 4             # [bird_y, velocity_y, pipe_dx, pipe_dy]
HIDDEN_DIM = 128          # Hidden layer size
BATCH_SIZE = 64           # Mini-batch size
MEMORY_SIZE = 100000      # Replay buffer capacity
GAMMA = 0.99              # Discount factor
LEARNING_RATE = 1e-3      # Learning rate
EPSILON_START = 1.0       # Initial exploration rate
EPSILON_MIN = 0.01        # Minimum exploration rate
EPSILON_DECAY = 0.9995    # Decay rate per episode
TAU = 0.005               # Soft update coefficient for target network
TRAIN_EVERY = 4           # Train every N steps
MIN_MEMORY = 1000         # Minimum transitions before training starts

# Rewards
REWARD_ALIVE = 0.1
REWARD_DEAD = -1.0
REWARD_PASS_PIPE = 1.0

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════
#  Network & Replay Buffer — shared by both agents
# ═══════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Simple MLP Q-Network.
    Input:  state (4 features)
    Output: Q-values for [no_jump, jump]
    """
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    """Experience replay buffer"""
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
#  DQN Agent — configurable as Vanilla or Double
# ═══════════════════════════════════════════════════════════════

class DQNAgent:
    """
    A single DQN agent. The `use_double` flag controls which algorithm it uses.
    Two instances of this class are created — one Vanilla, one Double —
    so they have completely independent networks, optimizers, and memories.
    """
    def __init__(self, name, use_double):
        self.name = name              # "Vanilla DQN" or "Double DQN"
        self.use_double = use_double  # The ONLY algorithmic difference

        # Each agent has its OWN networks (completely independent)
        self.online_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)

        # Each agent has its OWN replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)

        self.epsilon = EPSILON_START
        self.total_steps = 0

        # History for plotting
        self.score_history = []
        self.loss_history = []
        self.q_value_history = []

    def get_action(self, state, training=True):
        """ε-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_t = state.unsqueeze(0).to(device)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        """
        The ONLY difference between Vanilla and Double is inside this function.
        Everything else — network, memory, optimizer — is identical.
        """
        if len(self.memory) < max(BATCH_SIZE, MIN_MEMORY):
            return None, None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Current Q(s, a): what online_net thinks the chosen action is worth
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-value: what the action SHOULD be worth
        with torch.no_grad():
            if self.use_double:
                # ╔══════════════════════════════════════════════════════════╗
                # ║  DOUBLE DQN                                             ║
                # ║                                                         ║
                # ║  Step 1: online_net picks the best action for s'        ║
                # ║     a* = argmax_a Q_online(s', a)                       ║
                # ║                                                         ║
                # ║  Step 2: target_net evaluates that specific action      ║
                # ║     next_q = Q_target(s', a*)                           ║
                # ║                                                         ║
                # ║  WHY: online_net might overestimate action 1,           ║
                # ║  but target_net (different weights) likely won't        ║
                # ║  have the SAME bias → cross-check reduces inflation.   ║
                # ╚══════════════════════════════════════════════════════════╝
                best_actions = self.online_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            else:
                # ╔══════════════════════════════════════════════════════════╗
                # ║  VANILLA DQN                                            ║
                # ║                                                         ║
                # ║  target_net does BOTH: pick the max AND evaluate it     ║
                # ║     next_q = max_a Q_target(s', a)                      ║
                # ║                                                         ║
                # ║  PROBLEM: if Q_target(s', action=1) is noisy-high,     ║
                # ║  max will pick it → the high noise becomes the target  ║
                # ║  → online_net trains toward the inflated value         ║
                # ║  → Q-values systematically drift upward.               ║
                # ╚══════════════════════════════════════════════════════════╝
                next_q = self.target_net(next_states).max(1)[0]

            targets = rewards + GAMMA * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        avg_q = q_values.mean().item()
        return loss.item(), avg_q

    def update_target_network(self):
        """Soft update: target slowly tracks online"""
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def record_episode(self, score, avg_loss, avg_q):
        self.score_history.append(score)
        self.loss_history.append(avg_loss)
        self.q_value_history.append(avg_q)

    def save(self, filename=None):
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}_model.pth"
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'score_history': self.score_history,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history,
        }, filename)

    def load(self, filename=None):
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}_model.pth"
        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location=device, weights_only=False)
            self.online_net.load_state_dict(ckpt['online_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.epsilon = ckpt.get('epsilon', EPSILON_MIN)
            self.total_steps = ckpt.get('total_steps', 0)
            self.score_history = ckpt.get('score_history', [])
            self.loss_history = ckpt.get('loss_history', [])
            self.q_value_history = ckpt.get('q_value_history', [])
            print(f"[{self.name}] Model loaded from {filename}")
            return True
        return False


# ═══════════════════════════════════════════════════════════════
#  Comparison Plot — the payoff: two agents on one chart
# ═══════════════════════════════════════════════════════════════

def plot_comparison(vanilla_agent, double_agent, episode, filename="double_dqn_comparison.png"):
    """
    Plot both agents' training curves side by side.
    The Q-value subplot is the KEY chart — you should see Vanilla's Q-values
    climb higher than Double's, demonstrating overestimation.
    """
    if len(vanilla_agent.score_history) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Vanilla DQN vs Double DQN — Episode {episode}', fontsize=14, fontweight='bold')

    def smooth(data, window=50):
        if len(data) < 10:
            return np.array(data), np.arange(1, len(data) + 1)
        w = max(1, min(window, len(data) // 5))
        s = np.convolve(data, np.ones(w)/w, mode='valid')
        return s, np.arange(w, len(data) + 1)

    # ── Plot 1: Score comparison ──
    ax = axes[0]
    for agent, color, label in [(vanilla_agent, 'red', 'Vanilla DQN'),
                                 (double_agent, 'blue', 'Double DQN')]:
        if agent.score_history:
            ax.plot(agent.score_history, color=color, alpha=0.1, linewidth=0.5)
            s, e = smooth(agent.score_history)
            ax.plot(e, s, color=color, linewidth=2, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Score (higher = better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Q-value comparison (THE KEY CHART) ──
    ax = axes[1]
    for agent, color, label in [(vanilla_agent, 'red', 'Vanilla DQN'),
                                 (double_agent, 'blue', 'Double DQN')]:
        if agent.q_value_history:
            ax.plot(agent.q_value_history, color=color, alpha=0.1, linewidth=0.5)
            s, e = smooth(agent.q_value_history)
            ax.plot(e, s, color=color, linewidth=2, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Q-value')
    ax.set_title('Avg Q-value (Vanilla should be higher = overestimation!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Loss comparison ──
    ax = axes[2]
    for agent, color, label in [(vanilla_agent, 'red', 'Vanilla DQN'),
                                 (double_agent, 'blue', 'Double DQN')]:
        if agent.loss_history:
            ax.plot(agent.loss_history, color=color, alpha=0.1, linewidth=0.5)
            s, e = smooth(agent.loss_history)
            ax.plot(e, s, color=color, linewidth=2, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved to {filename}")


# ═══════════════════════════════════════════════════════════════
#  Game objects
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


# ═══════════════════════════════════════════════════════════════
#  Game — runs both agents in alternating episodes
# ═══════════════════════════════════════════════════════════════

class Game:
    """
    Training design:
      - Two independent agents: vanilla_agent and double_agent
      - They ALTERNATE episodes: episode 1 → Vanilla, episode 2 → Double, ...
      - Same game physics, same ε schedule, same hyperparams
      - The ONLY difference is the target Q computation
      - Display shows whichever agent is currently playing
      - Comparison plot overlays both agents' curves
    """
    def __init__(self, render=True):
        self.render_game = render
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - Vanilla DQN vs Double DQN")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 28)

        # ★ Two completely independent agents
        self.vanilla_agent = DQNAgent(name="Vanilla DQN", use_double=False)
        self.double_agent = DQNAgent(name="Double DQN", use_double=True)

        self.episode = 0
        self.training = True
        self.reset_game()

    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frame_count = 0
        self.ground_offset = 0
        for i in range(4):
            self.pipes.append(Pipe(SCREEN_WIDTH + i * PIPE_SPACING))

    def get_state(self):
        """4-dim normalized state vector"""
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

    def draw_info(self, agent):
        if not self.render_game:
            return

        is_double = agent.use_double
        name = agent.name
        color = (100, 200, 255) if is_double else (255, 150, 100)

        v_best = max(self.vanilla_agent.score_history) if self.vanilla_agent.score_history else 0
        d_best = max(self.double_agent.score_history) if self.double_agent.score_history else 0
        v_avg_q = self.vanilla_agent.q_value_history[-1] if self.vanilla_agent.q_value_history else 0
        d_avg_q = self.double_agent.q_value_history[-1] if self.double_agent.q_value_history else 0

        info_texts = [
            (f"Now Playing: {name}", color),
            (f"Episode: {self.episode}", WHITE),
            (f"Score: {int(self.score)}", WHITE),
            (f"Epsilon: {agent.epsilon:.4f}", WHITE),
            (f"Memory: {len(agent.memory)}", WHITE),
            (f"", WHITE),
            (f"--- Comparison ---", (200, 200, 200)),
            (f"Vanilla Best: {int(v_best)}", (255, 150, 100)),
            (f"Double  Best: {int(d_best)}", (100, 200, 255)),
            (f"Vanilla AvgQ: {v_avg_q:.3f}", (255, 150, 100)),
            (f"Double  AvgQ: {d_avg_q:.3f}", (100, 200, 255)),
        ]

        y = 10
        for text, c in info_texts:
            if text:
                surface = self.font_small.render(text, True, c)
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
                    self.vanilla_agent.save()
                    self.double_agent.save()
                    print("Both agents saved.")
                if event.key == pygame.K_l:
                    self.vanilla_agent.load()
                    self.double_agent.load()
                if event.key == pygame.K_r:
                    self.vanilla_agent = DQNAgent(name="Vanilla DQN", use_double=False)
                    self.double_agent = DQNAgent(name="Double DQN", use_double=True)
                    self.episode = 0
                    print("Both agents reset!")
        return True

    def run_episode(self, agent):
        """Run one episode with the given agent"""
        self.reset_game()
        state = self.get_state()
        episode_losses = []
        episode_q_values = []

        while True:
            if not self.handle_events():
                return None

            action = agent.get_action(state, training=self.training)
            reward, done = self.step(action)
            next_state = self.get_state()
            agent.total_steps += 1

            if self.training:
                agent.memory.push(state, action, reward, next_state, done)
                if agent.total_steps % TRAIN_EVERY == 0:
                    loss, avg_q = agent.train_step()
                    if loss is not None:
                        episode_losses.append(loss)
                        episode_q_values.append(avg_q)
                agent.update_target_network()

            state = next_state

            if self.render_game:
                self.draw_gradient_background()
                for pipe in self.pipes:
                    pipe.draw(self.screen)
                self.draw_ground()
                self.bird.draw(self.screen)
                self.draw_info(agent)
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
        return self.score, avg_loss, avg_q

    def run(self):
        print("=" * 58)
        print("  Flappy Bird — Vanilla DQN vs Double DQN (parallel)")
        print("=" * 58)
        print(f"Device: {device}")
        print()
        print("Design: Two independent agents alternate episodes.")
        print("  Odd episodes  → Vanilla DQN (red)")
        print("  Even episodes → Double DQN  (blue)")
        print("  Same game, same hyperparams, ONLY the target formula differs.")
        print()
        print("Controls:")
        print("  T - Toggle Training/Testing")
        print("  S - Save both agents")
        print("  L - Load both agents")
        print("  R - Reset both agents")
        print("  ESC - Quit")
        print("=" * 58)

        self.vanilla_agent.load()
        self.double_agent.load()

        while True:
            if not self.handle_events():
                break

            self.episode += 1

            # ★ Alternate: odd → Vanilla, even → Double
            if self.episode % 2 == 1:
                agent = self.vanilla_agent
            else:
                agent = self.double_agent

            result = self.run_episode(agent)
            if result is None:
                break

            score, avg_loss, avg_q = result

            if self.training:
                agent.record_episode(score, avg_loss, avg_q)
                agent.decay_epsilon()

            # Print progress every 50 episodes
            if self.episode % 50 == 0:
                v_scores = self.vanilla_agent.score_history[-25:]
                d_scores = self.double_agent.score_history[-25:]
                v_qs = self.vanilla_agent.q_value_history[-25:]
                d_qs = self.double_agent.q_value_history[-25:]

                v_avg = np.mean(v_scores) if v_scores else 0
                d_avg = np.mean(d_scores) if d_scores else 0
                v_q = np.mean(v_qs) if v_qs else 0
                d_q = np.mean(d_qs) if d_qs else 0

                print(f"Episode {self.episode}:")
                print(f"  Vanilla — AvgScore={v_avg:.0f}, AvgQ={v_q:.3f}, ε={self.vanilla_agent.epsilon:.4f}")
                print(f"  Double  — AvgScore={d_avg:.0f}, AvgQ={d_q:.3f}, ε={self.double_agent.epsilon:.4f}")

                if self.training:
                    plot_comparison(self.vanilla_agent, self.double_agent, self.episode)

            if self.episode % 200 == 0:
                self.vanilla_agent.save()
                self.double_agent.save()

        self.vanilla_agent.save()
        self.double_agent.save()
        plot_comparison(self.vanilla_agent, self.double_agent, self.episode)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()
