"""
Flappy Bird with Deep Q-Network (DQN)
Train an AI agent to play Flappy Bird using DQN with CNN
State: 6 consecutive frames (current + 5 previous), preprocessed to 80x80 grayscale
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
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 640
FPS = 60
TRAINING_FPS = 240

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
FRAME_STACK = 4           # Number of frames to stack (reduced for efficiency)
IMG_SIZE = 84             # Preprocessed image size (standard for DQN)
BATCH_SIZE = 64           # Mini-batch size for training (increased)
MEMORY_SIZE = 100000      # Replay memory capacity (increased)
GAMMA = 0.99              # Discount factor
LEARNING_RATE = 1e-3      # Learning rate (increased for faster learning)
EPSILON_START = 1.0       # Initial exploration rate
EPSILON_MIN = 0.05        # Minimum exploration rate (keep some exploration)
EPSILON_DECAY = 0.9995    # Decay rate per episode (even slower decay)
TARGET_UPDATE = 1         # Update target network every episode (for soft update)
TAU = 0.01                # Soft update coefficient (increased for faster adaptation)
REWARD_ALIVE = 0.1        # Reward for staying alive (small positive)
REWARD_DEAD = -1.0        # Penalty for dying (moderate)
REWARD_CLIP = 1.0         # Clip rewards to [-1, 1] range
FRAME_SKIP = 1            # No frame skip for more responsive control
TRAIN_EVERY = 2           # Train more frequently
MIN_MEMORY = 1000         # Minimum memory before training starts

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    """
    Deep Q-Network with CNN + MLP
    Input: (batch, FRAME_STACK, IMG_SIZE, IMG_SIZE) - stacked grayscale frames
    Output: (batch, 2) - Q-values for [no_jump, jump]
    """
    def __init__(self):
        super(DQN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate CNN output size dynamically
        # Input: 84x84 -> conv1(8,4): 20x20 -> conv2(4,2): 9x9 -> conv3(3,1): 7x7
        def conv_output_size(size, kernel, stride):
            return (size - kernel) // stride + 1
        
        size = IMG_SIZE
        size = conv_output_size(size, 8, 4)  # After conv1
        size = conv_output_size(size, 4, 2)  # After conv2
        size = conv_output_size(size, 3, 1)  # After conv3
        cnn_output_size = 64 * size * size
        
        # MLP layers
        self.fc1 = nn.Linear(cnn_output_size, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        # x shape: (batch, FRAME_STACK, IMG_SIZE, IMG_SIZE)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


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


class DQNAgent:
    def __init__(self):
        # Networks
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Exploration
        self.epsilon = EPSILON_START
        
        # Statistics
        self.episode = 0
        self.best_score = 0
        self.total_steps = 0
        
        # Training history for plotting
        self.loss_history = []      # Average loss per episode
        self.score_history = []     # Score per episode
        self.reward_history = []    # Total reward per episode
        self.epsilon_history = []   # Epsilon per episode
        
    def get_action(self, state, training=True):
        """
        Epsilon-greedy action selection
        state: (FRAME_STACK, IMG_SIZE, IMG_SIZE) tensor
        Returns: 0 (no jump) or 1 (jump)
        
        Training mode: epsilon-greedy (random exploration)
        Testing mode: greedy (always pick best action, epsilon=0)
        """
        # Only use epsilon exploration during training
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            # Greedy action selection (testing uses this path with epsilon=0 effectively)
            with torch.no_grad():
                state = state.unsqueeze(0).to(device)
                q_values = self.policy_net(state)
                return q_values.argmax(dim=1).item()
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < max(BATCH_SIZE, MIN_MEMORY):
            return None
            
        # Sample from replay memory
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Compute Q(s, a)
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use policy_net to select action, target_net to evaluate
        with torch.no_grad():
            # Policy net selects best action for next state
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Target net evaluates Q-value for that action
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            # Clip rewards to stabilize training
            clipped_rewards = torch.clamp(rewards, -REWARD_CLIP, REWARD_CLIP)
            target_q_values = clipped_rewards + GAMMA * next_q_values * (1 - dones)
        
        # Compute loss with MSE (more stable than Huber for small values)
        loss = F.mse_loss(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self, soft=True):
        """Update target network weights
        
        soft=True: Soft update with TAU coefficient (more stable)
        soft=False: Hard copy (original method)
        """
        if soft:
            # Soft update: target = TAU * policy + (1 - TAU) * target
            for target_param, policy_param in zip(self.target_net.parameters(), 
                                                   self.policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        
    def record_episode(self, score, avg_loss, episode_reward):
        """Record episode statistics for plotting"""
        self.score_history.append(score)
        self.loss_history.append(avg_loss)
        self.reward_history.append(episode_reward)
        self.epsilon_history.append(self.epsilon)
        
    def plot_training_curves(self, filename="dqn_training_curves.png"):
        """Plot and save training curves"""
        if len(self.loss_history) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'DQN Training Progress (Episode {self.episode})', fontsize=14, fontweight='bold')
        
        episodes = range(1, len(self.loss_history) + 1)
        
        # Plot 1: Loss curve
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.loss_history, 'b-', alpha=0.3, linewidth=0.5)
        # Moving average for smoothing
        if len(self.loss_history) >= 10:
            window = min(50, len(self.loss_history) // 5)
            if window >= 1:
                loss_smooth = np.convolve(self.loss_history, np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(self.loss_history) + 1), loss_smooth, 'b-', linewidth=2, label='Smoothed')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Score curve
        ax2 = axes[0, 1]
        ax2.plot(episodes, self.score_history, 'g-', alpha=0.3, linewidth=0.5)
        if len(self.score_history) >= 10:
            window = min(50, len(self.score_history) // 5)
            if window >= 1:
                score_smooth = np.convolve(self.score_history, np.ones(window)/window, mode='valid')
                ax2.plot(range(window, len(self.score_history) + 1), score_smooth, 'g-', linewidth=2, label='Smoothed')
        ax2.axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {int(self.best_score)}')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score')
        ax2.set_title('Episode Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Reward curve
        ax3 = axes[1, 0]
        ax3.plot(episodes, self.reward_history, 'orange', alpha=0.3, linewidth=0.5)
        if len(self.reward_history) >= 10:
            window = min(50, len(self.reward_history) // 5)
            if window >= 1:
                reward_smooth = np.convolve(self.reward_history, np.ones(window)/window, mode='valid')
                ax3.plot(range(window, len(self.reward_history) + 1), reward_smooth, 'orange', linewidth=2, label='Smoothed')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Total Reward')
        ax3.set_title('Episode Reward')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Epsilon curve
        ax4 = axes[1, 1]
        ax4.plot(episodes, self.epsilon_history, 'purple', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Exploration Rate (ε)')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved to {filename}")
        
    def save(self, filename="dqn_model.pth"):
        """Save model to file"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode,
            'best_score': self.best_score,
            'total_steps': self.total_steps,
            'loss_history': self.loss_history,
            'score_history': self.score_history,
            'reward_history': self.reward_history,
            'epsilon_history': self.epsilon_history
        }, filename)
        print(f"Model saved to {filename}")
        
    def load(self, filename="dqn_model.pth"):
        """Load model from file"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', EPSILON_MIN)
            self.episode = checkpoint.get('episode', 0)
            self.best_score = checkpoint.get('best_score', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.loss_history = checkpoint.get('loss_history', [])
            self.score_history = checkpoint.get('score_history', [])
            self.reward_history = checkpoint.get('reward_history', [])
            self.epsilon_history = checkpoint.get('epsilon_history', [])
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
            pygame.display.set_caption("Flappy Bird - DQN")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 24)
        else:
            # Offscreen surface for getting pixels
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            
        self.agent = DQNAgent()
        self.training = True
        
        # Frame buffer for stacking
        self.frame_buffer = deque(maxlen=FRAME_STACK)
        
        self.reset_game()
        
    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frame_count = 0
        self.game_over = False
        
        for i in range(4):
            self.pipes.append(Pipe(SCREEN_WIDTH + i * PIPE_SPACING))
            
        # Initialize frame buffer with empty frames
        self.frame_buffer.clear()
        initial_frame = self.get_preprocessed_frame()
        for _ in range(FRAME_STACK):
            self.frame_buffer.append(initial_frame)
            
    def preprocess_frame(self, frame):
        """
        Preprocess frame:
        1. Set sky region to black
        2. Resize to 80x80
        3. Convert to grayscale
        
        frame: pygame Surface
        Returns: (80, 80) numpy array normalized to [0, 1]
        """
        # Get pixel array (H, W, 3)
        pixels = pygame.surfarray.array3d(frame)
        pixels = np.transpose(pixels, (1, 0, 2))  # (W, H, 3) -> (H, W, 3)
        
        # Convert to grayscale first (better for edge detection)
        gray = np.dot(pixels[..., :3], [0.299, 0.587, 0.114])
        
        # Set sky region to black based on position and intensity
        # Sky is the darker gradient area (intensity < 80)
        # But keep pipes (brighter green), bird (bright yellow), ground (brown)
        sky_threshold = 85
        sky_mask = gray < sky_threshold
        gray[sky_mask] = 0
        
        # Crop to game area only (remove ground area for cleaner input)
        game_height = SCREEN_HEIGHT - GROUND_HEIGHT
        gray = gray[:game_height, :]
        
        # Resize using scipy-style approach (faster and better quality)
        h, w = gray.shape
        # Calculate scaling factors
        scale_h = IMG_SIZE / h
        scale_w = IMG_SIZE / w
        
        # Create output array
        resized = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                src_i = min(int(i / scale_h), h - 1)
                src_j = min(int(j / scale_w), w - 1)
                resized[i, j] = gray[src_i, src_j]
        
        # Normalize to [0, 1]
        resized = resized / 255.0
        
        return resized
    
    def save_debug_frame(self, filename="debug_frame.png"):
        """Save current preprocessed frame for debugging"""
        frame = self.get_preprocessed_frame()
        # Convert to 0-255 for saving
        frame_img = (frame * 255).astype(np.uint8)
        # Save using pygame
        surf = pygame.Surface((IMG_SIZE, IMG_SIZE))
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                val = frame_img[i, j]
                surf.set_at((j, i), (val, val, val))
        pygame.image.save(surf, filename)
        print(f"Debug frame saved to {filename}")
        
    def get_preprocessed_frame(self):
        """Render game and get preprocessed frame"""
        # Render to surface
        self.draw_game_only()
        
        # Preprocess
        frame = self.preprocess_frame(self.screen)
        
        return frame
        
    def get_stacked_state(self):
        """Get stacked frames as tensor"""
        # Stack frames: (6, 80, 80)
        stacked = np.stack(list(self.frame_buffer), axis=0)
        return torch.from_numpy(stacked).float()
        
    def draw_game_only(self):
        """Draw game without UI elements"""
        # Draw gradient background
        for y in range(SCREEN_HEIGHT - GROUND_HEIGHT):
            ratio = y / (SCREEN_HEIGHT - GROUND_HEIGHT)
            r = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * ratio)
            g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * ratio)
            b = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))
            
        # Draw pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)
            
        # Draw ground
        pygame.draw.rect(self.screen, GROUND_TOP, 
                        (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, 10))
        pygame.draw.rect(self.screen, GROUND_COLOR, 
                        (0, SCREEN_HEIGHT - GROUND_HEIGHT + 10, SCREEN_WIDTH, GROUND_HEIGHT - 10))
        
        # Draw bird
        self.bird.draw(self.screen)
        
    def draw_info(self):
        """Draw training information overlay"""
        if not self.render_game:
            return
            
        info_texts = [
            f"Episode: {self.agent.episode}",
            f"Score: {int(self.score)}",
            f"Best: {int(self.agent.best_score)}",
            f"Epsilon: {self.agent.epsilon:.4f}",
            f"Memory: {len(self.agent.memory)}/{MEMORY_SIZE}",
            f"Steps: {self.agent.total_steps}",
            f"Mode: {'Training' if self.training else 'Testing'}",
            f"Device: {device}"
        ]
        
        y = 10
        for text in info_texts:
            surface = self.font_small.render(text, True, WHITE)
            shadow = self.font_small.render(text, True, BLACK)
            self.screen.blit(shadow, (12, y + 1))
            self.screen.blit(surface, (10, y))
            y += 20
            
        # Instructions
        instructions = [
            "T: Toggle Train/Test",
            "S: Save Model",
            "L: Load Model", 
            "R: Reset Agent",
            "D: Debug Frame",
            "ESC: Quit"
        ]
        y = SCREEN_HEIGHT - GROUND_HEIGHT - len(instructions) * 18 - 10
        for text in instructions:
            surface = self.font_small.render(text, True, (180, 180, 180))
            self.screen.blit(surface, (10, y))
            y += 18
            
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
        
    def get_next_pipe(self):
        """Get the next pipe that bird needs to pass"""
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                return pipe
        return self.pipes[-1] if self.pipes else None
        
    def step(self, action):
        """
        Execute one game step
        Returns: (reward, done)
        """
        if action == 1:
            self.bird.jump()
            
        self.bird.update()
        
        for pipe in self.pipes:
            pipe.update(BIRD_SPEED_X)
            
        # Check if bird passed a pipe (bonus reward)
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
            
            # Reward shaping: reward for being close to gap center
            next_pipe = self.get_next_pipe()
            if next_pipe:
                gap_center = next_pipe.gap_y
                distance_to_center = abs(self.bird.y - gap_center)
                max_distance = PIPE_GAP / 2
                
                # Normalize distance: 0 (at center) to 1 (at edge of gap)
                normalized_dist = min(distance_to_center / max_distance, 1.0)
                
                # Reward: +0.1 at center, -0.1 at edge
                position_reward = 0.1 * (1 - 2 * normalized_dist)
                reward += position_reward
            
            # Bonus reward for passing a pipe
            if passed_pipe:
                reward += 1.0
                
            self.score += 1
            
        self.frame_count += 1
        
        return reward, done
        
    def handle_events(self):
        """Handle pygame events, returns False if should quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_t:
                    self.training = not self.training
                    if self.training:
                        print(f"Mode: Training (epsilon={self.agent.epsilon:.4f})")
                    else:
                        print(f"Mode: Testing (epsilon=0, pure greedy policy)")
                if event.key == pygame.K_s:
                    self.agent.save()
                if event.key == pygame.K_l:
                    self.agent.load()
                if event.key == pygame.K_r:
                    self.agent = DQNAgent()
                    print("Agent reset!")
                if event.key == pygame.K_d:
                    self.save_debug_frame()
        return True
        
    def run_episode(self):
        """Run one complete episode"""
        self.reset_game()
        
        state = self.get_stacked_state()
        episode_reward = 0
        episode_loss = []
        frame_skip_counter = 0
        current_action = 0
        
        while True:
            if not self.handle_events():
                return None
            
            # Frame skipping: only select new action every FRAME_SKIP frames
            is_decision_frame = (frame_skip_counter == 0)
            if is_decision_frame:
                current_action = self.agent.get_action(state, training=self.training)
            frame_skip_counter = (frame_skip_counter + 1) % max(1, FRAME_SKIP)
            
            # Execute action (repeat same action during frame skip)
            reward, done = self.step(current_action)
            episode_reward += reward
            self.agent.total_steps += 1
            
            # Get next state
            next_frame = self.get_preprocessed_frame()
            self.frame_buffer.append(next_frame)
            next_state = self.get_stacked_state()
            
            # Store transition and train (only on decision frames to avoid duplicates)
            if self.training and is_decision_frame:
                self.agent.memory.push(state, current_action, reward, next_state, done)
                
                # Train every TRAIN_EVERY steps
                if self.agent.total_steps % TRAIN_EVERY == 0:
                    loss = self.agent.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                    
            state = next_state
            
            # Render
            if self.render_game:
                self.draw_info()
                pygame.display.flip()
                if self.training:
                    self.clock.tick(TRAINING_FPS)
                else:
                    self.clock.tick(FPS)
                    
            if done:
                break
                
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        return self.score, avg_loss, episode_reward
        
    def run(self):
        """Main training loop"""
        print("=" * 50)
        print("Flappy Bird DQN Training")
        print("=" * 50)
        print(f"Device: {device}")
        print(f"Frame stack: {FRAME_STACK}")
        print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
        print("Controls:")
        print("  T - Toggle Training/Testing mode")
        print("  S - Save model")
        print("  L - Load model")
        print("  R - Reset agent")
        print("  ESC - Quit")
        print("=" * 50)
        
        # Try to load existing model
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
                
            score, avg_loss, episode_reward = result
            
            # Record episode statistics
            if self.training:
                self.agent.record_episode(score, avg_loss, episode_reward)
            
            # Update best score
            if score > self.agent.best_score:
                self.agent.best_score = score
                print(f"New best score: {int(score)} (Episode {self.agent.episode})")
                
            # Decay epsilon
            if self.training:
                self.agent.decay_epsilon()
                
            # Update target network
            if self.training and self.agent.episode % TARGET_UPDATE == 0:
                self.agent.update_target_network()
                print(f"Target network updated (Episode {self.agent.episode})")
                
            # Print progress
            if self.agent.episode % 50 == 0:
                print(f"Episode {self.agent.episode}: Score={int(score)}, "
                      f"Best={int(self.agent.best_score)}, "
                      f"Reward={episode_reward:.2f}, "
                      f"Epsilon={self.agent.epsilon:.4f}, "
                      f"Loss={avg_loss:.4f}, "
                      f"Memory={len(self.agent.memory)}")
                # Plot training curves every 50 episodes
                if self.training:
                    self.agent.plot_training_curves()
                      
            # Auto-save
            if self.agent.episode % 200 == 0:
                self.agent.save()
                
        # Save on exit
        self.agent.save()
        self.agent.plot_training_curves()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()

