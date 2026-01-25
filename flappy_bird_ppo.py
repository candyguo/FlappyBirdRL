"""
Flappy Bird with PPO (Proximal Policy Optimization)

Policy: pi(a|s) from Actor
Value:  V(s) from Critic
Uses clipped surrogate objective + GAE advantages.
"""

import pygame
import random
import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
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

# PPO Hyperparameters
STATE_DIM = 4
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 2048

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.actor = nn.Linear(HIDDEN_DIM, 2)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def act(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


class PPOAgent:
    def __init__(self):
        self.model = ActorCritic().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        # Stats
        self.episode = 0
        self.best_score = 0
        self.score_history = []
        self.loss_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def compute_gae(self, next_value):
        rewards = self.rewards
        values = self.values + [next_value]
        dones = self.dones
        # GAE backward recursion (t from T-1 to 0):
        # δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        # A_t = δ_t + γλ A_{t+1}
        #
        # t=4: A4 = δ4
        # t=3: A3 = δ3 + γλ A4
        # t=2: A2 = δ2 + γλ A3
        # t=1: A1 = δ1 + γλ A2
        # t=0: A0 = δ0 + γλ A1

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def update(self, next_value):
        if not self.states:
            return 0.0, 0.0, 0.0

        advantages, returns = self.compute_gae(next_value)

        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        print("old_log_probs shape: ", torch.stack(self.log_probs).shape)
        old_log_probs = torch.stack(self.log_probs).detach().squeeze()  # Fix: squeeze to [N]
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_actor = 0.0
        total_critic = 0.0

        dataset_size = states.size(0)
        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns[batch_idx]
                b_adv = advantages[batch_idx]

                new_log_probs, entropy, values = self.model.evaluate(b_states, b_actions)
                values = values.squeeze(-1)
                print("new_log_probs shape: ", new_log_probs.shape)

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_adv
                
                # PPO clip objective: maximize min(ratio*A, clip(ratio)*A)
                # To minimize, we take negative
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(values, b_returns)
                entropy_bonus = entropy.mean()

                # Total loss: minimize (-policy_objective + value_loss - entropy_bonus)
                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss += loss.item()
                total_actor += actor_loss.item()
                total_critic += critic_loss.item()

        steps = max(1, (dataset_size // MINI_BATCH_SIZE) * PPO_EPOCHS)
        avg_loss = total_loss / steps
        avg_actor = total_actor / steps
        avg_critic = total_critic / steps

        self.clear()
        return avg_loss, avg_actor, avg_critic

    def record_episode(self, score, loss, actor_loss, critic_loss):
        self.score_history.append(score)
        self.loss_history.append(loss)
        self.actor_loss_history.append(actor_loss)
        self.critic_loss_history.append(critic_loss)

    def plot_training_curves(self, filename="ppo_training_curves.png"):
        if len(self.score_history) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"PPO Training (Episode {self.episode})", fontsize=14, fontweight="bold")

        episodes = range(1, len(self.score_history) + 1)

        # Score
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.score_history, "g-", alpha=0.3, linewidth=0.5)
        if len(self.score_history) >= 10:
            window = min(50, len(self.score_history) // 5)
            if window >= 1:
                smooth = np.convolve(self.score_history, np.ones(window) / window, mode="valid")
                ax1.plot(range(window, len(self.score_history) + 1), smooth, "g-", linewidth=2, label="Smoothed")
        ax1.axhline(y=self.best_score, color="r", linestyle="--", label=f"Best: {int(self.best_score)}")
        ax1.set_title("Episode Score")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Total loss
        ax2 = axes[0, 1]
        ax2.plot(episodes, self.loss_history, "b-", alpha=0.3, linewidth=0.5)
        ax2.set_title("Total Loss")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)

        # Actor loss
        ax3 = axes[1, 0]
        ax3.plot(episodes, self.actor_loss_history, "purple", alpha=0.3, linewidth=0.5)
        ax3.set_title("Actor Loss")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Loss")
        ax3.grid(True, alpha=0.3)

        # Critic loss
        ax4 = axes[1, 1]
        ax4.plot(episodes, self.critic_loss_history, "orange", alpha=0.3, linewidth=0.5)
        ax4.set_title("Critic Loss")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Loss")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Training curves saved to {filename}")

    def save(self, filename="ppo_model.pth"):
        torch.save({
            "model": self.model.state_dict(),
            "episode": self.episode,
            "best_score": self.best_score,
            "score_history": self.score_history,
            "loss_history": self.loss_history,
            "actor_loss_history": self.actor_loss_history,
            "critic_loss_history": self.critic_loss_history,
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename="ppo_model.pth"):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.model.load_state_dict(checkpoint["model"])
            self.episode = checkpoint.get("episode", 0)
            self.best_score = checkpoint.get("best_score", 0)
            self.score_history = checkpoint.get("score_history", [])
            self.loss_history = checkpoint.get("loss_history", [])
            self.actor_loss_history = checkpoint.get("actor_loss_history", [])
            self.critic_loss_history = checkpoint.get("critic_loss_history", [])
            print(f"Model loaded from {filename}")
            return True
        return False


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
            (self.x + self.radius - 5, self.y + 10),
        ]
        pygame.draw.polygon(screen, BIRD_BEAK, beak_points)

    def get_rect(self):
        return pygame.Rect(
            self.x - self.radius + 5,
            self.y - self.radius + 5,
            self.radius * 2 - 10,
            self.radius * 2 - 10,
        )


class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(
            MIN_PIPE_HEIGHT + PIPE_GAP // 2,
            SCREEN_HEIGHT - GROUND_HEIGHT - MIN_PIPE_HEIGHT - PIPE_GAP // 2,
        )
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
        bottom_rect = pygame.Rect(
            self.x - 5,
            bottom_y,
            PIPE_WIDTH + 10,
            SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y,
        )
        return top_rect, bottom_rect

    def is_offscreen(self):
        return self.x + PIPE_WIDTH < 0


class Game:
    def __init__(self, render=True):
        self.render_game = render
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - PPO")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 24)
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        self.agent = PPOAgent()
        self.training = True
        self.reset_game()

    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frame_count = 0

        for i in range(4):
            self.pipes.append(Pipe(SCREEN_WIDTH + i * PIPE_SPACING))

    def get_next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                return pipe
        return self.pipes[-1] if self.pipes else None

    def get_state(self):
        next_pipe = self.get_next_pipe()
        if next_pipe:
            pipe_dx = (next_pipe.x - self.bird.x) / SCREEN_WIDTH
            pipe_dy = (self.bird.y - next_pipe.gap_y) / SCREEN_HEIGHT
        else:
            pipe_dx = 1.0
            pipe_dy = 0.0

        return [
            self.bird.y / SCREEN_HEIGHT,
            self.bird.velocity_y / 15.0,
            pipe_dx,
            pipe_dy,
        ]

    def draw_gradient_background(self):
        for y in range(SCREEN_HEIGHT - GROUND_HEIGHT):
            ratio = y / (SCREEN_HEIGHT - GROUND_HEIGHT)
            r = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * ratio)
            g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * ratio)
            b = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

    def draw_ground(self):
        pygame.draw.rect(self.screen, GROUND_TOP, (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, 10))
        pygame.draw.rect(
            self.screen,
            GROUND_COLOR,
            (0, SCREEN_HEIGHT - GROUND_HEIGHT + 10, SCREEN_WIDTH, GROUND_HEIGHT - 10),
        )

    def draw_info(self):
        if not self.render_game:
            return

        info_texts = [
            f"Episode: {self.agent.episode}",
            f"Score: {int(self.score)}",
            f"Best: {int(self.agent.best_score)}",
            f"Mode: {'Training' if self.training else 'Testing'}",
        ]

        y = 10
        for text in info_texts:
            surface = self.font_small.render(text, True, WHITE)
            shadow = self.font_small.render(text, True, BLACK)
            self.screen.blit(shadow, (12, y + 1))
            self.screen.blit(surface, (10, y))
            y += 20

        instructions = [
            "T: Toggle Train/Test",
            "S: Save Model",
            "L: Load Model",
            "R: Reset Agent",
            "ESC: Quit",
        ]
        y = SCREEN_HEIGHT - GROUND_HEIGHT - len(instructions) * 18 - 10
        for text in instructions:
            surface = self.font_small.render(text, True, (180, 180, 180))
            self.screen.blit(surface, (10, y))
            y += 18

    def draw(self):
        if not self.render_game:
            return

        self.draw_gradient_background()
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.draw_ground()
        self.bird.draw(self.screen)
        self.draw_info()

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

        # Check passed pipe
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
        
        # Reward shaping (same as AC for fair comparison)
        if done:
            reward = -1.0
        else:
            reward = 0.1
            if passed_pipe:
                reward += 1.0

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
                    if self.training:
                        print("Mode: Training")
                    else:
                        print("Mode: Testing (greedy policy)")
                if event.key == pygame.K_s:
                    self.agent.save()
                if event.key == pygame.K_l:
                    self.agent.load()
                if event.key == pygame.K_r:
                    self.agent = PPOAgent()
                    print("Agent reset!")
        return True

    def run_episode(self):
        self.reset_game()
        done = False

        while not done:
            if not self.handle_events():
                return None

            state = self.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _, _ = self.agent.model.act(state_tensor)

            reward, done = self.step(action.item())

            self.draw()
            if self.render_game:
                pygame.display.flip()
                self.clock.tick(FPS)

        return self.score

    def run_rollout(self, rollout_steps):
        """Collect fixed number of steps across episodes."""
        print(f"Starting rollout for {rollout_steps} steps...", flush=True)
        steps = 0
        last_value = 0.0
        last_done = False

        while steps < rollout_steps:
            if steps % 500 == 0:
                print(f"  step {steps}/{rollout_steps}", flush=True)
            if not self.handle_events():
                return None

            state = self.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, _, value = self.agent.model.act(state_tensor)

            reward, done = self.step(action.item())

            self.agent.store(
                state,
                action.item(),
                log_prob.detach(),
                reward,
                value.item(),
                float(done),
            )

            last_value = value.item()
            last_done = done
            steps += 1

            self.draw()
            if self.render_game:
                pygame.display.flip()
                self.clock.tick(TRAINING_FPS)

            if done:
                # Episode finished; record stats and reset.
                self.agent.episode += 1
                self.agent.record_episode(self.score, 0.0, 0.0, 0.0)
                if self.score > self.agent.best_score:
                    self.agent.best_score = self.score
                    print(f"New best score: {int(self.score)} (Episode {self.agent.episode})")
                self.reset_game()

        return 0.0 if last_done else last_value

    def run(self):
        print("=" * 50)
        print("Flappy Bird - PPO")
        print("=" * 50)
        print(f"Device: {device}")
        print(f"State dim: {STATE_DIM}")
        print(f"Hidden dim: {HIDDEN_DIM}")
        print("Controls:")
        print("  T - Toggle Training/Testing mode")
        print("  S - Save model")
        print("  L - Load model")
        print("  R - Reset agent")
        print("  ESC - Quit")
        print("=" * 50)

        self.agent.load()

        running = True
        print("Entering main loop...", flush=True)
        while running:
            if not self.handle_events():
                break

            if self.training:
                # Collect fixed rollout steps and update PPO
                next_value = self.run_rollout(ROLLOUT_STEPS)
                if next_value is None:
                    running = False
                    break
               
                print("Updating model...", flush=True)
                loss, actor_loss, critic_loss = self.agent.update(next_value=next_value)
                # Update last recorded episode losses
                if self.agent.score_history:
                    self.agent.loss_history[-1] = loss
                    self.agent.actor_loss_history[-1] = actor_loss
                    self.agent.critic_loss_history[-1] = critic_loss

                if self.agent.episode % 50 == 0 and self.agent.episode > 0:
                    avg_score = np.mean(self.agent.score_history[-50:]) if self.agent.score_history else 0
                    print(
                        f"Episode {self.agent.episode}: "
                        f"Best={int(self.agent.best_score)}, "
                        f"Avg(50)={avg_score:.1f}, "
                        f"Loss={loss:.4f}, "
                        f"Actor={actor_loss:.4f}, "
                        f"Critic={critic_loss:.4f}"
                    )
                    self.agent.plot_training_curves()

                if self.agent.episode % 200 == 0 and self.agent.episode > 0:
                    self.agent.save()
            else:
                # Testing: run a full episode with greedy policy
                self.agent.episode += 1
                score = self.run_episode()
                if score is None:
                    running = False
                    break
                if score > self.agent.best_score:
                    self.agent.best_score = score
                    print(f"New best score: {int(score)} (Episode {self.agent.episode})")

        self.agent.save()
        self.agent.plot_training_curves()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()
