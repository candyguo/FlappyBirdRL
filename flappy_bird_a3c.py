"""
Flappy Bird with A3C (Asynchronous Advantage Actor-Critic)

Key features:
1. Multiple parallel workers (async training)
2. Shared global network
3. Each worker has local network copy
4. Asynchronous gradient updates

Reference: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
"""

import pygame
import random
import sys
import os
import numpy as np
from collections import deque
import time
import threading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Game constants
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 640
FPS = 60

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

# A3C Hyperparameters
STATE_DIM = 4              # [bird_y, velocity_y, pipe_dx, pipe_dy]
HIDDEN_DIM = 128           # Hidden layer size
LEARNING_RATE = 1e-4       # Learning rate (lower for stability with async)
GAMMA = 0.99               # Discount factor
ENTROPY_COEF = 0.01        # Entropy bonus coefficient
VALUE_LOSS_COEF = 0.5      # Value loss coefficient
MAX_GRAD_NORM = 40.0       # Gradient clipping
T_MAX = 20                 # Steps before update (n-step returns)
NUM_WORKERS = 4            # Number of parallel workers

# Device (A3C typically uses CPU for simplicity with multiprocessing)
device = torch.device("cpu")


class ActorCriticNetwork(nn.Module):
    """
    Shared Actor-Critic Network
    Actor head: outputs action probabilities π(a|s)
    Critic head: outputs state value V(s)
    """
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        
        # Actor head (policy)
        self.actor = nn.Linear(HIDDEN_DIM, 2)
        
        # Critic head (value)
        self.critic = nn.Linear(HIDDEN_DIM, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Smaller init for output layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor: action probabilities
        policy = F.softmax(self.actor(x), dim=-1)
        
        # Critic: state value
        value = self.critic(x)
        
        return policy, value


class SharedRMSprop(optim.RMSprop):
    """
    RMSprop optimizer with shared state for multiprocessing
    More stable than SharedAdam for A3C
    """
    def __init__(self, params, lr=1e-4, alpha=0.99, eps=1e-8):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps)
        
        # Share optimizer state across processes
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['square_avg'] = torch.zeros_like(p.data)
                state['square_avg'].share_memory_()


class FlappyBirdEnv:
    """
    Flappy Bird environment (no rendering for workers)
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.bird_x = 80
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_vy = 0
        self.bird_radius = 20
        
        self.pipes = []
        for i in range(4):
            self.pipes.append({
                'x': SCREEN_WIDTH + i * PIPE_SPACING,
                'gap_y': random.randint(MIN_PIPE_HEIGHT + PIPE_GAP // 2,
                                        SCREEN_HEIGHT - GROUND_HEIGHT - MIN_PIPE_HEIGHT - PIPE_GAP // 2),
                'passed': False
            })
            
        self.score = 0
        self.done = False
        return self.get_state()
        
    def get_next_pipe(self):
        for pipe in self.pipes:
            if pipe['x'] + PIPE_WIDTH > self.bird_x:
                return pipe
        return self.pipes[-1] if self.pipes else None
        
    def get_state(self):
        next_pipe = self.get_next_pipe()
        
        if next_pipe:
            pipe_dx = (next_pipe['x'] - self.bird_x) / SCREEN_WIDTH
            pipe_dy = (self.bird_y - next_pipe['gap_y']) / SCREEN_HEIGHT
        else:
            pipe_dx = 1.0
            pipe_dy = 0.0
            
        return np.array([
            self.bird_y / SCREEN_HEIGHT,
            self.bird_vy / 15.0,
            pipe_dx,
            pipe_dy
        ], dtype=np.float32)
        
    def step(self, action):
        if action == 1:
            self.bird_vy = JUMP_STRENGTH
            
        self.bird_vy += GRAVITY
        self.bird_y += self.bird_vy
        
        for pipe in self.pipes:
            pipe['x'] -= BIRD_SPEED_X
            
        # Check passed pipe
        passed_pipe = False
        for pipe in self.pipes:
            if not pipe['passed'] and pipe['x'] + PIPE_WIDTH < self.bird_x:
                pipe['passed'] = True
                passed_pipe = True
                
        # Remove offscreen pipes and add new ones
        if self.pipes and self.pipes[0]['x'] + PIPE_WIDTH < 0:
            self.pipes.pop(0)
            new_x = self.pipes[-1]['x'] + PIPE_SPACING
            self.pipes.append({
                'x': new_x,
                'gap_y': random.randint(MIN_PIPE_HEIGHT + PIPE_GAP // 2,
                                        SCREEN_HEIGHT - GROUND_HEIGHT - MIN_PIPE_HEIGHT - PIPE_GAP // 2),
                'passed': False
            })
            
        # Check collision
        done = self._check_collision()
        
        # Reward
        if done:
            reward = -1.0
        else:
            reward = 0.1
            if passed_pipe:
                reward += 1.0
                
        self.score += 1
        self.done = done
        
        return self.get_state(), reward, done
        
    def _check_collision(self):
        # Ground/ceiling collision
        if self.bird_y + self.bird_radius > SCREEN_HEIGHT - GROUND_HEIGHT:
            return True
        if self.bird_y - self.bird_radius < 0:
            return True
            
        # Pipe collision
        bird_rect = (self.bird_x - self.bird_radius + 5,
                     self.bird_y - self.bird_radius + 5,
                     self.bird_radius * 2 - 10,
                     self.bird_radius * 2 - 10)
        
        for pipe in self.pipes:
            top_height = pipe['gap_y'] - PIPE_GAP // 2
            bottom_y = pipe['gap_y'] + PIPE_GAP // 2
            
            # Top pipe
            if self._rect_collision(bird_rect, 
                                    (pipe['x'] - 5, 0, PIPE_WIDTH + 10, top_height)):
                return True
            # Bottom pipe
            if self._rect_collision(bird_rect,
                                    (pipe['x'] - 5, bottom_y, PIPE_WIDTH + 10,
                                     SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y)):
                return True
                
        return False
        
    def _rect_collision(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)


def ensure_shared_grads(local_model, global_model):
    """
    Copy gradients from local model to global model
    """
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            return
        global_param._grad = local_param.grad


class Worker(mp.Process):
    """
    A3C Worker process
    """
    def __init__(self, worker_id, global_model, optimizer, global_episode, 
                 global_scores, lock, max_episodes=10000):
        super(Worker, self).__init__()
        
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.global_episode = global_episode
        self.global_scores = global_scores
        self.lock = lock
        self.max_episodes = max_episodes
        
        # Local model
        self.local_model = ActorCriticNetwork()
        
        # Environment
        self.env = FlappyBirdEnv()
        
    def run(self):
        """Worker main loop"""
        while self.global_episode.value < self.max_episodes:
            # Sync local model with global model
            self.local_model.load_state_dict(self.global_model.state_dict())
            
            # Run episode
            score = self._run_episode()
            
            # Update global counter and scores
            with self.lock:
                self.global_episode.value += 1
                episode = self.global_episode.value
                
                # Store score
                if len(self.global_scores) < 1000:
                    self.global_scores.append(score)
                else:
                    self.global_scores[episode % 1000] = score
                    
            # Print progress
            if episode % 100 == 0:
                recent_scores = list(self.global_scores)[-100:]
                avg_score = np.mean(recent_scores) if recent_scores else 0
                max_score = max(recent_scores) if recent_scores else 0
                print(f"Worker {self.worker_id} | Episode {episode} | "
                      f"Score: {int(score)} | Avg(100): {avg_score:.1f} | "
                      f"Max: {int(max_score)}")
                      
    def _run_episode(self):
        """Run one episode with n-step updates"""
        state = self.env.reset()
        done = False
        
        while not done:
            # Collect T_MAX steps
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            entropies = []
            
            for _ in range(T_MAX):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action from local model
                policy, value = self.local_model(state_tensor)
                
                dist = Categorical(policy)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                # Store
                states.append(state)
                actions.append(action.item())
                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                # Step environment
                next_state, reward, done = self.env.step(action.item())
                rewards.append(reward)
                
                state = next_state
                
                if done:
                    break
                    
            # Compute returns and advantages
            self._update(states, actions, rewards, values, log_probs, entropies, 
                        done, next_state)
                        
        return self.env.score
        
    def _update(self, states, actions, rewards, values, log_probs, entropies, 
                done, last_state):
        """
        Compute n-step returns and update global model
        """
        # Bootstrap value
        R = 0.0
        if not done:
            state_tensor = torch.FloatTensor(last_state).unsqueeze(0)
            _, value = self.local_model(state_tensor)
            R = value.item()
            
        # Compute returns backwards
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Stack tensors
        log_probs = torch.stack(log_probs).squeeze()
        values = torch.stack(values).squeeze()
        entropies = torch.stack(entropies).squeeze()
        
        # Handle scalar case
        if log_probs.dim() == 0:
            log_probs = log_probs.unsqueeze(0)
            values = values.unsqueeze(0)
            entropies = entropies.unsqueeze(0)
            returns = returns.unsqueeze(0)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Compute losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        # Total loss
        loss = policy_loss + VALUE_LOSS_COEF * value_loss + ENTROPY_COEF * entropy_loss
        
        # Backprop on local model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), MAX_GRAD_NORM)
        
        # Copy gradients to global model and update
        ensure_shared_grads(self.local_model, self.global_model)
        self.optimizer.step()


class A3CAgent:
    """
    A3C Agent manager
    """
    def __init__(self, num_workers=NUM_WORKERS):
        self.num_workers = num_workers
        
        # Global model (shared memory)
        self.global_model = ActorCriticNetwork()
        self.global_model.share_memory()
        
        # Shared optimizer
        self.optimizer = SharedRMSprop(self.global_model.parameters(), lr=LEARNING_RATE)
        
        # Shared counters
        self.global_episode = mp.Value('i', 0)
        self.manager = mp.Manager()
        self.global_scores = self.manager.list()
        self.lock = mp.Lock()
        
        # Statistics
        self.best_score = 0
        
    def train(self, max_episodes=10000):
        """Start distributed training"""
        print("=" * 60)
        print("A3C Training - Flappy Bird")
        print("=" * 60)
        print(f"Number of workers: {self.num_workers}")
        print(f"Max episodes: {max_episodes}")
        print(f"T_MAX (steps per update): {T_MAX}")
        print(f"Learning rate: {LEARNING_RATE}")
        print("=" * 60)
        
        # Load if exists
        self.load()
        
        # Create workers
        workers = []
        for i in range(self.num_workers):
            worker = Worker(i, self.global_model, self.optimizer,
                          self.global_episode, self.global_scores,
                          self.lock, max_episodes)
            workers.append(worker)
            
        # Start workers
        for worker in workers:
            worker.start()
            
        # Wait for completion
        for worker in workers:
            worker.join()
            
        # Save final model
        self.save()
        self.plot_training_curves()
        
        print("Training complete!")
        
    def get_action(self, state, greedy=True):
        """Get action from global model"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = self.global_model(state_tensor)
            
        if greedy:
            return policy.argmax(dim=1).item()
        else:
            dist = Categorical(policy)
            return dist.sample().item()
            
    def save(self, filename="a3c_model.pth"):
        """Save model"""
        scores = list(self.global_scores)
        torch.save({
            'model': self.global_model.state_dict(),
            'episode': self.global_episode.value,
            'scores': scores,
            'best_score': max(scores) if scores else 0
        }, filename)
        print(f"Model saved to {filename}")
        
    def load(self, filename="a3c_model.pth"):
        """Load model"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.global_model.load_state_dict(checkpoint['model'])
            self.global_episode.value = checkpoint.get('episode', 0)
            scores = checkpoint.get('scores', [])
            for s in scores:
                self.global_scores.append(s)
            self.best_score = checkpoint.get('best_score', 0)
            print(f"Model loaded from {filename}")
            return True
        return False
        
    def plot_training_curves(self, filename="a3c_training_curves.png"):
        """Plot training curves"""
        scores = list(self.global_scores)
        if len(scores) < 2:
            return
            
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f'A3C Training ({len(scores)} episodes)', fontsize=14, fontweight='bold')
        
        ax.plot(scores, 'g-', alpha=0.3, linewidth=0.5)
        
        # Smoothed curve
        if len(scores) >= 50:
            window = min(100, len(scores) // 5)
            score_smooth = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(scores)), score_smooth, 'g-', linewidth=2, label='Smoothed')
            
        best = max(scores)
        ax.axhline(y=best, color='r', linestyle='--', label=f'Best: {int(best)}')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Episode Score')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved to {filename}")


# ===== Game Visualization =====

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


class GameVisualizer:
    """
    Visualize trained A3C agent playing
    """
    def __init__(self, agent):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird - A3C Testing")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.agent = agent
        self.reset_game()
        
    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        
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
            pipe_dy
        ]
        
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
                        
    def draw_info(self, episode, best_score):
        info_texts = [
            f"Episode: {episode}",
            f"Score: {int(self.score)}",
            f"Best: {int(best_score)}",
            "Mode: A3C Testing"
        ]
        
        y = 10
        for text in info_texts:
            surface = self.font_small.render(text, True, WHITE)
            shadow = self.font_small.render(text, True, BLACK)
            self.screen.blit(shadow, (12, y + 1))
            self.screen.blit(surface, (10, y))
            y += 20
            
        instructions = [
            "ESC: Quit"
        ]
        y = SCREEN_HEIGHT - GROUND_HEIGHT - len(instructions) * 18 - 10
        for text in instructions:
            surface = self.font_small.render(text, True, (180, 180, 180))
            self.screen.blit(surface, (10, y))
            y += 18
            
    def draw(self, episode, best_score):
        self.draw_gradient_background()
        
        for pipe in self.pipes:
            pipe.draw(self.screen)
            
        self.draw_ground()
        self.bird.draw(self.screen)
        self.draw_info(episode, best_score)
        
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
            
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
            
        if self.pipes and self.pipes[0].is_offscreen():
            self.pipes.pop(0)
            new_x = self.pipes[-1].x + PIPE_SPACING
            self.pipes.append(Pipe(new_x))
            
        done = self.check_collision()
        self.score += 1
        
        return done
        
    def run(self):
        """Run visualization"""
        print("=" * 50)
        print("A3C Testing Mode")
        print("=" * 50)
        
        episode = 0
        best_score = 0
        running = True
        
        while running:
            self.reset_game()
            episode += 1
            
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                            
                if not running:
                    break
                    
                state = self.get_state()
                action = self.agent.get_action(state, greedy=True)
                done = self.step(action)
                
                self.draw(episode, best_score)
                pygame.display.flip()
                self.clock.tick(FPS)
                
                if done:
                    if self.score > best_score:
                        best_score = self.score
                        print(f"New best score: {int(best_score)}")
                    break
                    
            if episode % 10 == 0:
                print(f"Episode {episode}: Score = {int(self.score)}")
                
        pygame.quit()


def main():
    """Main function"""
    print("=" * 60)
    print("Flappy Bird - A3C (Asynchronous Advantage Actor-Critic)")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Train new model")
    print("  2. Continue training")
    print("  3. Test trained model")
    print("  4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    agent = A3CAgent(num_workers=NUM_WORKERS)
    
    if choice == '1':
        # Fresh training
        if os.path.exists("a3c_model.pth"):
            os.remove("a3c_model.pth")
        max_eps = int(input("Max episodes (default 5000): ").strip() or "5000")
        agent.train(max_episodes=max_eps)
        
    elif choice == '2':
        # Continue training
        max_eps = int(input("Additional episodes (default 2000): ").strip() or "2000")
        agent.load()
        current = agent.global_episode.value
        agent.train(max_episodes=current + max_eps)
        
    elif choice == '3':
        # Test mode
        if agent.load():
            visualizer = GameVisualizer(agent)
            visualizer.run()
        else:
            print("No trained model found. Please train first.")
            
    else:
        print("Goodbye!")
        

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method('spawn', force=True)
    main()

