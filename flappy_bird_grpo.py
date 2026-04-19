"""
Flappy Bird with GRPO (Group Relative Policy Optimization)

From DeepSeek-R1 (2024). A simpler alternative to PPO that removes the critic.

Core idea:
    PPO:   advantage = reward - V(s)         ← needs critic network
    GRPO:  advantage = (R - mean) / std      ← needs only a GROUP of trajectories

    For each training step:
      1. Sample G trajectories using current policy
      2. Get each trajectory's total reward R_i
      3. Normalize within the group:
           advantage_i = (R_i - mean(R)) / std(R)
      4. Update policy with PPO-style clipped objective + KL penalty

    No critic. No value function. No GAE.
    Just: "did this trajectory do better or worse than its peers?"

Why this works:
    - Critic V(s) is just an estimate of expected reward — often inaccurate
    - Group mean IS the expected reward (empirical estimate from actual samples)
    - Simpler, fewer hyperparameters, often works just as well

Why it fits Flappy Bird perfectly:
    - We can cheaply run many episodes (game is fast)
    - Each episode gives a clear scalar score
    - No need for per-step value estimation

Comparison:
    PPO:   1 trajectory → needs V(s) for each step → critic introduces bias
    GRPO:  G trajectories → compare scores → group statistics are unbiased
"""

import pygame
import random
import sys
import os
import numpy as np
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

# ─── GRPO Hyperparameters ────────────────────────────────────
STATE_DIM = 4
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
GAMMA = 0.99               # Discount factor for per-step rewards
CLIP_EPS = 0.2             # PPO-style clipping
KL_BETA = 0.04             # KL penalty coefficient (prevents policy collapse)
ENTROPY_COEF = 0.01        # Entropy bonus for exploration
MAX_GRAD_NORM = 0.5
GROUP_SIZE = 8              # G: number of trajectories per group
UPDATE_EPOCHS = 4           # PPO epochs per update
MINI_BATCH_SIZE = 256       # Mini-batch size for gradient updates
MAX_STEPS_PER_EPISODE = 2000

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════
#  Policy Network (Actor only — no critic!)
# ═══════════════════════════════════════════════════════════════
#
#  PPO has: Actor (policy) + Critic (value function)
#  GRPO has: Actor ONLY — the whole point is to remove the critic

class PolicyNetwork(nn.Module):
    """
    Actor only. No value head — GRPO doesn't need one.

    Compare with PPO's network:
        PPO:   shared → policy_head + value_head   (2 heads)
        GRPO:  shared → policy_head                 (1 head)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(HIDDEN_DIM, 2)   # [P(no_jump), P(jump)]

    def forward(self, x):
        features = self.net(x)
        logits = self.policy_head(features)
        return logits

    def get_action(self, state):
        """Sample action, return (action, log_prob)."""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions):
        """Evaluate (state, action) pairs for policy update."""
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


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
        top_rect = pygame.Rect(self.x - 5, 0, PIPE_WIDTH + 10, top_height)
        bottom_rect = pygame.Rect(self.x - 5, bottom_y, PIPE_WIDTH + 10,
                                  SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y)
        return top_rect, bottom_rect

    def get_bottom_rect_pos(self):
        bottom_y = self.gap_y + PIPE_GAP // 2
        return self.x - 5, bottom_y

    def is_offscreen(self):
        return self.x + PIPE_WIDTH < 0


class GameEnv:
    """Headless game environment for fast data collection."""
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
#  GRPO Agent
# ═══════════════════════════════════════════════════════════════

class GRPOAgent:
    def __init__(self):
        self.policy = PolicyNetwork().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

        # Reference policy for KL penalty (frozen copy of initial policy)
        self.ref_policy = copy.deepcopy(self.policy).to(device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.episode = 0
        self.best_score = 0

        self.score_history = []
        self.loss_history = []
        self.kl_history = []
        self.group_std_history = []    # Track group score diversity

    def collect_group(self, env):
        """
        ★ GRPO Step 1: Sample G trajectories using current policy.

        Each trajectory = one full episode of Flappy Bird.
        Returns G trajectories with their states, actions, log_probs, and total scores.

        PPO comparison:
            PPO:  collect 1 long rollout, compute V(s) for each step
            GRPO: collect G short episodes, compare their total scores
        """
        group = []

        for _ in range(GROUP_SIZE):
            states = []
            actions = []
            log_probs = []

            state = env.reset()

            for _ in range(MAX_STEPS_PER_EPISODE):
                state_t = state.unsqueeze(0).to(device)
                with torch.no_grad():
                    action, log_prob = self.policy.get_action(state_t)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)

                state, done = env.step(action)
                if done:
                    break

            trajectory = {
                'states': torch.stack(states),
                'actions': torch.tensor(actions, dtype=torch.long),
                'log_probs': torch.cat(log_probs).detach(),
                'score': env.score,
            }
            group.append(trajectory)

        return group

    def compute_group_advantages(self, group):
        """
        ★ GRPO Step 2: Compute advantages using group-relative normalization.

        This is the CORE of GRPO — replacing the critic with group statistics.

            PPO advantage:    A_t = R_t - V(s_t)          ← needs critic
            GRPO advantage:   A_i = (R_i - mean) / std    ← needs only the group

        Every step in trajectory i gets the SAME advantage (trajectory-level).

        In DeepSeek's LLM version:
            - Each "trajectory" = one generated response
            - Score = reward model score for that response
            - All tokens in the response share the same advantage

        Here:
            - Each "trajectory" = one Flappy Bird episode
            - Score = game score (how many frames survived)
            - All steps in the episode share the same advantage
        """
        scores = [t['score'] for t in group]
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-8   # prevent division by zero

        for traj in group:
            # ★ The key formula — no critic needed:
            advantage = (traj['score'] - mean_score) / std_score
            traj['advantage'] = advantage

        return mean_score, std_score

    def compute_kl_penalty(self, states):
        """
        KL divergence between current policy and reference policy.
        Same role as in RLHF: prevent the policy from drifting too far.

        Without KL: policy might collapse to always-jump or never-jump.
        """
        logits_new = self.policy(states)
        with torch.no_grad():
            logits_ref = self.ref_policy(states)

        log_probs_new = F.log_softmax(logits_new, dim=-1)
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)
        probs_new = log_probs_new.exp()

        # KL(π_new || π_ref) = Σ_a π_new(a) × [log π_new(a) - log π_ref(a)]
        kl = (probs_new * (log_probs_new - log_probs_ref)).sum(-1)
        return kl

    def update(self, group):
        """
        ★ GRPO Step 3: Policy update with clipped objective + KL penalty.

        For each trajectory i in the group:
            advantage_i = (score_i - group_mean) / group_std
            All steps in trajectory i share this advantage.

        Then PPO-style clipped update:
            ratio = π_new(a|s) / π_old(a|s)
            L = -min(ratio × A, clip(ratio) × A) + β × KL

        The only difference from PPO is HOW the advantage is computed:
            PPO:  per-step, from critic V(s)
            GRPO: per-trajectory, from group statistics
        """
        # Flatten all trajectories into one batch
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        for traj in group:
            n = len(traj['states'])
            all_states.append(traj['states'])
            all_actions.append(traj['actions'])
            all_old_log_probs.append(traj['log_probs'])
            # ★ Same advantage for every step in this trajectory
            all_advantages.append(torch.full((n,), traj['advantage']))

        states = torch.cat(all_states).to(device)
        actions = torch.cat(all_actions).to(device)
        old_log_probs = torch.cat(all_old_log_probs).to(device)
        advantages = torch.cat(all_advantages).to(device)

        # PPO-style update epochs
        total_loss = 0
        total_kl = 0
        num_updates = 0

        for _ in range(UPDATE_EPOCHS):
            indices = torch.randperm(len(states))

            for start in range(0, len(states), MINI_BATCH_SIZE):
                idx = indices[start:start + MINI_BATCH_SIZE]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_lp = old_log_probs[idx]
                mb_adv = advantages[idx]

                # Current policy log-probs
                new_log_probs, entropy = self.policy.evaluate(mb_states, mb_actions)

                # PPO clipped ratio
                ratio = (new_log_probs - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL penalty (replaces critic's role in stabilization)
                kl = self.compute_kl_penalty(mb_states)
                kl_loss = KL_BETA * kl.mean()

                # Entropy bonus
                entropy_loss = -ENTROPY_COEF * entropy.mean()

                # Total loss = policy + KL + entropy
                loss = policy_loss + kl_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss += loss.item()
                total_kl += kl.mean().item()
                num_updates += 1

        avg_loss = total_loss / max(num_updates, 1)
        avg_kl = total_kl / max(num_updates, 1)
        return avg_loss, avg_kl

    def update_reference(self):
        """Periodically update reference policy to current policy."""
        self.ref_policy.load_state_dict(self.policy.state_dict())

    def record_episode(self, scores, avg_loss, avg_kl, group_std):
        best = max(scores)
        self.score_history.append(best)
        self.loss_history.append(avg_loss)
        self.kl_history.append(avg_kl)
        self.group_std_history.append(group_std)
        if best > self.best_score:
            self.best_score = best

    def plot_training_curves(self, filename="grpo_training_curves.png"):
        if len(self.score_history) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'GRPO Training (Episode {self.episode})',
                    fontsize=14, fontweight='bold')

        episodes = range(1, len(self.score_history) + 1)

        def smooth(data, window=20):
            if len(data) < 5:
                return data, range(1, len(data) + 1)
            w = max(1, min(window, len(data) // 5))
            s = np.convolve(data, np.ones(w)/w, mode='valid')
            return s, range(w, len(data) + 1)

        # 1: Score
        ax = axes[0, 0]
        ax.plot(episodes, self.score_history, 'g-', alpha=0.3, linewidth=0.5)
        s, e = smooth(self.score_history)
        ax.plot(e, s, 'g-', linewidth=2, label='Smoothed')
        ax.axhline(y=self.best_score, color='r', linestyle='--', alpha=0.5,
                   label=f'Best: {int(self.best_score)}')
        ax.set_title('Best Group Score')
        ax.set_xlabel('Update')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2: Loss
        ax = axes[0, 1]
        ax.plot(episodes, self.loss_history, 'b-', alpha=0.3, linewidth=0.5)
        s, e = smooth(self.loss_history)
        ax.plot(e, s, 'b-', linewidth=2, label='Smoothed')
        ax.set_title('Policy Loss')
        ax.set_xlabel('Update')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3: KL divergence
        ax = axes[1, 0]
        ax.plot(episodes, self.kl_history, 'red', alpha=0.3, linewidth=0.5)
        s, e = smooth(self.kl_history)
        ax.plot(e, s, 'red', linewidth=2, label='Smoothed')
        ax.set_title(f'KL(π || π_ref)  [β={KL_BETA}]')
        ax.set_xlabel('Update')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4: Group score std (diversity)
        ax = axes[1, 1]
        ax.plot(episodes, self.group_std_history, 'purple', alpha=0.3, linewidth=0.5)
        s, e = smooth(self.group_std_history)
        ax.plot(e, s, 'purple', linewidth=2, label='Smoothed')
        ax.set_title(f'Group Score Std (G={GROUP_SIZE})\n(diversity of outcomes)')
        ax.set_xlabel('Update')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved to {filename}")

    def save(self, filename="grpo_model.pth"):
        torch.save({
            'policy': self.policy.state_dict(),
            'ref_policy': self.ref_policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'best_score': self.best_score,
            'score_history': self.score_history,
            'loss_history': self.loss_history,
            'kl_history': self.kl_history,
            'group_std_history': self.group_std_history,
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename="grpo_model.pth"):
        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location=device, weights_only=False)
            self.policy.load_state_dict(ckpt['policy'])
            self.ref_policy.load_state_dict(ckpt['ref_policy'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.episode = ckpt.get('episode', 0)
            self.best_score = ckpt.get('best_score', 0)
            self.score_history = ckpt.get('score_history', [])
            self.loss_history = ckpt.get('loss_history', [])
            self.kl_history = ckpt.get('kl_history', [])
            self.group_std_history = ckpt.get('group_std_history', [])
            print(f"Model loaded from {filename}")
            return True
        return False


# ═══════════════════════════════════════════════════════════════
#  Visual Game
# ═══════════════════════════════════════════════════════════════

class Game:
    def __init__(self, render=True):
        self.render_game = render
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - GRPO")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 28)

        self.agent = GRPOAgent()
        self.env = GameEnv()
        self.training = True
        self.reset_game()

    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
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
            ("Algorithm: GRPO (no critic!)", (255, 200, 50)),
            (f"Update: {self.agent.episode}", WHITE),
            (f"Score: {int(self.score)}", WHITE),
            (f"Best: {int(self.agent.best_score)}", WHITE),
            (f"Group size: {GROUP_SIZE}", WHITE),
            (f"KL beta: {KL_BETA}", (150, 200, 255)),
            (f"Mode: {'Training' if self.training else 'Testing'}", WHITE),
        ]
        if self.agent.kl_history:
            info_texts.append((f"KL: {self.agent.kl_history[-1]:.4f}", (255, 150, 150)))
        if self.agent.group_std_history:
            info_texts.append((f"Group std: {self.agent.group_std_history[-1]:.1f}", (200, 150, 255)))

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
        done = self.check_collision()
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
                if event.key == pygame.K_t:
                    self.training = not self.training
                    print(f"Mode: {'Training' if self.training else 'Testing'}")
                if event.key == pygame.K_s:
                    self.agent.save()
                if event.key == pygame.K_l:
                    self.agent.load()
                if event.key == pygame.K_r:
                    self.agent = GRPOAgent()
                    print("Agent reset!")
        return True

    def run_visual_episode(self):
        """Play one visual episode with current policy."""
        self.reset_game()
        state = self.get_state()

        while True:
            if not self.handle_events():
                return None

            state_t = state.unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = self.agent.policy.get_action(state_t)

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
                if self.training:
                    if TRAINING_FPS > 0:
                        self.clock.tick(TRAINING_FPS)
                else:
                    self.clock.tick(FPS)

            if done:
                break
        return self.score

    def run(self):
        print("=" * 55)
        print("  Flappy Bird — GRPO")
        print("  (Group Relative Policy Optimization)")
        print("=" * 55)
        print(f"Device: {device}")
        print(f"Group size: {GROUP_SIZE}")
        print(f"KL beta: {KL_BETA}")
        print("Controls:")
        print("  T - Toggle Training / Testing")
        print("  S - Save   L - Load   R - Reset")
        print("  ESC - Quit")
        print("=" * 55)

        self.agent.load()

        while True:
            if not self.handle_events():
                break

            if self.training:
                self.agent.episode += 1

                # ★ GRPO training loop — the complete algorithm:
                # Step 1: Sample a group of G trajectories
                group = self.agent.collect_group(self.env)

                # Step 2: Compute group-relative advantages
                mean_score, std_score = self.agent.compute_group_advantages(group)

                # Step 3: PPO-style policy update
                avg_loss, avg_kl = self.agent.update(group)

                # Update reference policy periodically
                if self.agent.episode % 20 == 0:
                    self.agent.update_reference()

                # Record
                scores = [t['score'] for t in group]
                self.agent.record_episode(scores, avg_loss, avg_kl, std_score)

                if self.agent.episode % 10 == 0:
                    print(f"  [GRPO] Update {self.agent.episode}: "
                          f"Scores={[int(s) for s in sorted(scores)]}, "
                          f"Mean={mean_score:.0f}, Std={std_score:.0f}, "
                          f"Best={int(self.agent.best_score)}, "
                          f"KL={avg_kl:.4f}")

                if self.agent.episode % 50 == 0:
                    self.agent.plot_training_curves()
                    self.agent.save()

            # Show a visual episode
            score = self.run_visual_episode()
            if score is None:
                break

        self.agent.save()
        self.agent.plot_training_curves()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()
