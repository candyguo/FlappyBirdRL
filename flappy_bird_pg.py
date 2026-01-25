"""
Flappy Bird with Policy Gradient (REINFORCE)
Train an AI agent using pure policy gradient method
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

# Policy Gradient Hyperparameters
STATE_DIM = 4             # [bird_y, velocity_y, pipe_dx, pipe_dy]
HIDDEN_DIM = 128          # Hidden layer size
LEARNING_RATE = 1e-3      # Learning rate
GAMMA = 0.99              # Discount factor
ENTROPY_COEF = 0.01       # Entropy bonus for exploration

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PolicyNetwork(nn.Module):
    """
    Policy Network: outputs action probabilities
    Input: state (4 features)
    Output: [P(no_jump), P(jump)]
    """
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 2)  # 2 actions: no_jump, jump
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)


class PolicyGradientAgent:
    def __init__(self):
        self.policy_net = PolicyNetwork().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
        
        # Statistics
        self.episode = 0
        self.best_score = 0
        self.score_history = []
        self.loss_history = []
        
    def get_action(self, state, training=True):
        """
        Sample action from policy
        Returns: action (0 or 1), log_prob
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.policy_net(state_tensor)
        
        if training:
            # Sample from distribution
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            self.log_probs.append(log_prob)
            return action.item()
        else:
            # Greedy: pick most probable action
            return probs.argmax(dim=1).item()
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
        
    def compute_returns(self):
        """
        Compute discounted returns G_t = sum(gamma^k * r_{t+k})
        """
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Normalize returns (reduces variance)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self):
        """
        REINFORCE update:
        loss = -sum(log π(a|s) * G_t)
        """
        print(f"[DEBUG] update() called, rewards count: {len(self.rewards)}")
        
        if len(self.rewards) == 0:
            print("[DEBUG] rewards is empty, skipping update")
            return 0
            
        returns = self.compute_returns()
        
        # Compute policy loss
        policy_loss = []
        entropy = 0

        print(f"[DEBUG] returns: {returns[:5]}... (showing first 5)")
        print(f"[DEBUG] log_probs count: {len(self.log_probs)}")
        
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # Add entropy bonus for exploration
        # (Note: we compute approximate entropy from stored log_probs)
        
        loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode storage
        loss_value = loss.item()
        self.log_probs = []
        self.rewards = []
        
        return loss_value
    
    def record_episode(self, score, loss):
        """Record episode statistics"""
        self.score_history.append(score)
        self.loss_history.append(loss)
        
    def plot_training_curves(self, filename="pg_training_curves.png"):
        """Plot and save training curves"""
        if len(self.score_history) < 2:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Policy Gradient Training (Episode {self.episode})', fontsize=14, fontweight='bold')
        
        episodes = range(1, len(self.score_history) + 1)
        
        # Plot 1: Score
        ax1 = axes[0]
        ax1.plot(episodes, self.score_history, 'g-', alpha=0.3, linewidth=0.5)
        if len(self.score_history) >= 10:
            window = min(50, len(self.score_history) // 5)
            if window >= 1:
                score_smooth = np.convolve(self.score_history, np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(self.score_history) + 1), score_smooth, 'g-', linewidth=2, label='Smoothed')
        ax1.axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {int(self.best_score)}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Episode Score')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Loss
        ax2 = axes[1]
        ax2.plot(episodes, self.loss_history, 'b-', alpha=0.3, linewidth=0.5)
        if len(self.loss_history) >= 10:
            window = min(50, len(self.loss_history) // 5)
            if window >= 1:
                loss_smooth = np.convolve(self.loss_history, np.ones(window)/window, mode='valid')
                ax2.plot(range(window, len(self.loss_history) + 1), loss_smooth, 'b-', linewidth=2, label='Smoothed')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Policy Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved to {filename}")
        
    def save(self, filename="pg_model.pth"):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'best_score': self.best_score,
            'score_history': self.score_history,
            'loss_history': self.loss_history
        }, filename)
        print(f"Model saved to {filename}")
        
    def load(self, filename="pg_model.pth"):
        """Load model"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.episode = checkpoint.get('episode', 0)
            self.best_score = checkpoint.get('best_score', 0)
            self.score_history = checkpoint.get('score_history', [])
            self.loss_history = checkpoint.get('loss_history', [])
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
    
    def is_offscreen(self):
        return self.x + PIPE_WIDTH < 0


class Game:
    def __init__(self, render=True):
        self.render_game = render
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - Policy Gradient")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 24)
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            
        self.agent = PolicyGradientAgent()
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
        """Get the next pipe that bird needs to pass"""
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                return pipe
        return self.pipes[-1] if self.pipes else None
            
    def get_state(self):
        """
        Get state vector:
        [bird_y (normalized), velocity_y (normalized), 
         horizontal_dist_to_pipe (normalized), vertical_dist_to_gap (normalized)]
        """
        next_pipe = self.get_next_pipe()
        
        if next_pipe:
            pipe_dx = (next_pipe.x - self.bird.x) / SCREEN_WIDTH
            pipe_dy = (self.bird.y - next_pipe.gap_y) / SCREEN_HEIGHT
        else:
            pipe_dx = 1.0
            pipe_dy = 0.0
            
        state = [
            self.bird.y / SCREEN_HEIGHT,           # Normalized bird y
            self.bird.velocity_y / 15.0,           # Normalized velocity
            pipe_dx,                                # Normalized horizontal distance
            pipe_dy                                 # Normalized vertical distance to gap
        ]
        
        return state
    
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
            f"Episode: {self.agent.episode}",
            f"Score: {int(self.score)}",
            f"Best: {int(self.agent.best_score)}",
            f"Mode: {'Training' if self.training else 'Testing'}"
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
            "ESC: Quit"
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
        """Execute one game step"""
        if action == 1:
            self.bird.jump()
            
        self.bird.update()
        
        for pipe in self.pipes:
            pipe.update(BIRD_SPEED_X)
            
        # Check if passed pipe
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
        
        # Reward design
        if done:
            reward = -1.0
        else:
            reward = 0.1  # Small reward for staying alive
            if passed_pipe:
                reward += 1.0  # Bonus for passing pipe
            
        self.score += 1
        self.frame_count += 1
        
        return reward, done
        
    def handle_events(self):
        """Handle pygame events"""
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
                    self.agent = PolicyGradientAgent()
                    print("Agent reset!")
        return True
        
    def run_episode(self):
        """Run one complete episode"""
        self.reset_game()
        
        while True:
            if not self.handle_events():
                return None
                
            state = self.get_state()
            action = self.agent.get_action(state, training=self.training)
            reward, done = self.step(action)
            
            if self.training:
                self.agent.store_reward(reward)
            
            # Render
            self.draw()
            if self.render_game:
                pygame.display.flip()
                if self.training:
                    self.clock.tick(TRAINING_FPS)
                else:
                    self.clock.tick(FPS)
                    
            if done:
                break
                
        # Update policy at end of episode (REINFORCE updates after full episode)
        loss = 0
        print(f"[DEBUG] Episode ended, training={self.training}, score={self.score}")
        if self.training:
            loss = self.agent.update()
        else:
            print("[DEBUG] Not training, skipping update")
            
        return self.score, loss
        
    def run(self):
        """Main training loop"""
        print("=" * 50)
        print("Flappy Bird - Policy Gradient (REINFORCE)")
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
        while running:
            if not self.handle_events():
                break
                
            self.agent.episode += 1
            result = self.run_episode()
            
            if result is None:
                running = False
                break
                
            score, loss = result
            
            # Record stats
            if self.training:
                self.agent.record_episode(score, loss)
            
            # Update best score
            if score > self.agent.best_score:
                self.agent.best_score = score
                print(f"New best score: {int(score)} (Episode {self.agent.episode})")
                
            # Print progress
            if self.agent.episode % 50 == 0:
                avg_score = np.mean(self.agent.score_history[-50:]) if self.agent.score_history else 0
                print(f"Episode {self.agent.episode}: Score={int(score)}, "
                      f"Best={int(self.agent.best_score)}, "
                      f"Avg(50)={avg_score:.1f}, "
                      f"Loss={loss:.4f}")
                if self.training:
                    self.agent.plot_training_curves()
                      
            # Auto-save
            if self.agent.episode % 200 == 0:
                self.agent.save()
                
        self.agent.save()
        self.agent.plot_training_curves()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()

