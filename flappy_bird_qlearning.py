"""
Flappy Bird with Q-Learning
Train an AI agent to play Flappy Bird using Q-learning algorithm
"""

import pygame
import random
import sys
import pickle
import os
from collections import defaultdict

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 640
FPS = 60
TRAINING_FPS = 240  # Fast but not unlimited, allows event handling

# Colors - Retro arcade theme
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
SCORE_COLOR = (255, 255, 255)

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

# Q-Learning parameters
LEARNING_RATE = 0.1      # α - learning rate
DISCOUNT_FACTOR = 0.95   # γ - discount factor
EPSILON_START = 1.0      # Initial exploration rate
EPSILON_MIN = 0.01       # Minimum exploration rate
EPSILON_DECAY = 0.9995   # Decay rate per episode

# Rewards
REWARD_ALIVE = 1
REWARD_DEAD = -1000

# State discretization bins
X_BINS = 20   # Discretize x distance into bins
Y_BINS = 20   # Discretize y distance into bins
VY_BINS = 10  # Discretize velocity into bins


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
        # Apply gravity
        self.velocity_y += GRAVITY
        self.y += self.velocity_y
        
    def draw(self, screen):
        # Draw bird body
        pygame.draw.circle(screen, BIRD_COLOR, (int(self.x), int(self.y)), self.radius)
        
        # Draw eye
        eye_x = self.x + 8
        eye_y = self.y - 5
        pygame.draw.circle(screen, WHITE, (int(eye_x), int(eye_y)), 7)
        pygame.draw.circle(screen, BIRD_EYE, (int(eye_x + 2), int(eye_y)), 4)
        
        # Draw beak
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
        # Top pipe
        top_height = self.gap_y - PIPE_GAP // 2
        pygame.draw.rect(screen, PIPE_COLOR, (self.x, 0, PIPE_WIDTH, top_height))
        pygame.draw.rect(screen, PIPE_HIGHLIGHT, (self.x, 0, 8, top_height))
        pygame.draw.rect(screen, PIPE_SHADOW, (self.x + PIPE_WIDTH - 8, 0, 8, top_height))
        pygame.draw.rect(screen, PIPE_COLOR, (self.x - 5, top_height - 30, PIPE_WIDTH + 10, 30))
        
        # Bottom pipe
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
        """Get the position of bottom pipe's top-left corner"""
        bottom_y = self.gap_y + PIPE_GAP // 2
        return self.x - 5, bottom_y
    
    def is_offscreen(self):
        return self.x + PIPE_WIDTH < 0


class QLearningAgent:
    def __init__(self):
        # Q-table: maps (state, action) -> Q-value
        self.q_table = defaultdict(float)
        self.epsilon = EPSILON_START
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        
        # Statistics
        self.episode = 0
        self.best_score = 0
        self.scores_history = []
        
    def discretize_state(self, dx, dy, velocity_y):
        """
        Discretize continuous state into discrete bins
        dx: horizontal distance to next pipe
        dy: vertical distance to bottom pipe's top edge (gap bottom)
        velocity_y: bird's vertical velocity
        """
        # Discretize x distance (0 to PIPE_SPACING)
        dx_bin = int(max(0, min(X_BINS - 1, dx * X_BINS / PIPE_SPACING)))
        
        # Discretize y distance (-SCREEN_HEIGHT to SCREEN_HEIGHT)
        dy_normalized = (dy + SCREEN_HEIGHT) / (2 * SCREEN_HEIGHT)
        dy_bin = int(max(0, min(Y_BINS - 1, dy_normalized * Y_BINS)))
        
        # Discretize velocity (-15 to 15)
        vy_normalized = (velocity_y + 15) / 30
        vy_bin = int(max(0, min(VY_BINS - 1, vy_normalized * VY_BINS)))
        
        return (dx_bin, dy_bin, vy_bin)
    
    def get_action(self, state, training=True):
        """
        Epsilon-greedy action selection
        Returns: 0 (no jump) or 1 (jump)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, 1)
        else:
            # Exploit: choose best action
            q_no_jump = self.q_table[(state, 0)]
            q_jump = self.q_table[(state, 1)]
            
            if q_jump > q_no_jump:
                return 1
            elif q_no_jump > q_jump:
                return 0
            else:
                return random.randint(0, 1)
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update rule:
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        """
        current_q = self.q_table[(state, action)]
        
        if done:
            # Terminal state
            max_next_q = 0
        else:
            # Get max Q-value for next state
            max_next_q = max(self.q_table[(next_state, 0)], 
                           self.q_table[(next_state, 1)])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[(state, action)] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        
    def save(self, filename="q_table.pkl"):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'episode': self.episode,
                'best_score': self.best_score
            }, f)
        print(f"Model saved to {filename}")
        
    def load(self, filename="q_table.pkl"):
        """Load Q-table from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(float, data['q_table'])
                self.epsilon = data.get('epsilon', EPSILON_MIN)
                self.episode = data.get('episode', 0)
                self.best_score = data.get('best_score', 0)
            print(f"Model loaded from {filename}")
            return True
        return False


class Game:
    def __init__(self, render=True):
        self.render_game = render
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - Q-Learning")
            self.clock = pygame.time.Clock()
            self.font_large = pygame.font.Font(None, 72)
            self.font_medium = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 28)
        
        self.agent = QLearningAgent()
        self.training = False
        self.reset_game()
        
    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frame_count = 0
        self.game_over = False
        self.ground_offset = 0
        
        # Generate initial pipes
        for i in range(4):
            self.pipes.append(Pipe(SCREEN_WIDTH + i * PIPE_SPACING))
            
    def get_state(self):
        """Get current state: distance to next pipe's bottom rect"""
        # Find next pipe (the one bird hasn't passed yet)
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            next_pipe = self.pipes[-1]
        
        # Get bottom rect position
        pipe_x, pipe_y = next_pipe.get_bottom_rect_pos()
        
        # Calculate distances
        dx = pipe_x - self.bird.x  # Horizontal distance to pipe
        dy = self.bird.y - pipe_y   # Vertical distance to gap bottom (negative = above)
        
        # Discretize state
        return self.agent.discretize_state(dx, dy, self.bird.velocity_y)
    
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
        """Draw training information"""
        info_texts = [
            f"Episode: {self.agent.episode}",
            f"Score: {int(self.score)}",
            f"Best: {int(self.agent.best_score)}",
            f"Epsilon: {self.agent.epsilon:.4f}",
            f"Q-table size: {len(self.agent.q_table)}",
            f"Mode: {'Training' if self.training else 'Testing'}"
        ]
        
        y = 10
        for text in info_texts:
            surface = self.font_small.render(text, True, WHITE)
            shadow = self.font_small.render(text, True, BLACK)
            self.screen.blit(shadow, (12, y + 2))
            self.screen.blit(surface, (10, y))
            y += 25
            
        # Draw instructions
        instructions = [
            "T: Toggle Training/Testing",
            "S: Save Model",
            "L: Load Model",
            "R: Reset Agent",
            "ESC: Quit"
        ]
        y = SCREEN_HEIGHT - GROUND_HEIGHT - len(instructions) * 22 - 10
        for text in instructions:
            surface = self.font_small.render(text, True, (180, 180, 180))
            self.screen.blit(surface, (10, y))
            y += 22
            
    def check_collision(self):
        bird_rect = self.bird.get_rect()
        
        # Check ground collision
        if self.bird.y + self.bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT:
            return True
            
        # Check ceiling collision
        if self.bird.y - self.bird.radius < 0:
            return True
            
        # Check pipe collision
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.get_rects()
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                return True
                
        return False
        
    def step(self, action):
        """
        Execute one game step
        Returns: (next_state, reward, done)
        """
        # Execute action
        if action == 1:
            self.bird.jump()
            
        # Update game
        self.bird.update()
        
        for pipe in self.pipes:
            pipe.update(BIRD_SPEED_X)
            
        # Remove offscreen pipes and add new ones
        if self.pipes and self.pipes[0].is_offscreen():
            self.pipes.pop(0)
            new_x = self.pipes[-1].x + PIPE_SPACING
            self.pipes.append(Pipe(new_x))
            
        # Check collision
        done = self.check_collision()
        
        if done:
            reward = REWARD_DEAD
        else:
            reward = REWARD_ALIVE
            self.score += 1
            
        self.frame_count += 1
        next_state = self.get_state()
        
        return next_state, reward, done
        
    def draw(self):
        if not self.render_game:
            return
            
        self.draw_gradient_background()
        
        for pipe in self.pipes:
            pipe.draw(self.screen)
            
        self.draw_ground()
        self.bird.draw(self.screen)
        self.draw_info()
        
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
                    print(f"Mode: {'Training' if self.training else 'Testing'}")
                if event.key == pygame.K_s:
                    self.agent.save()
                if event.key == pygame.K_l:
                    self.agent.load()
                if event.key == pygame.K_r:
                    self.agent = QLearningAgent()
                    print("Agent reset!")
        return True
        
    def run_episode(self):
        """Run one complete episode"""
        self.reset_game()
        state = self.get_state()
        total_reward = 0
        
        while True:
            # Handle pygame events
            if not self.handle_events():
                return None
            
            
            # Get action from agent
            action = self.agent.get_action(state, training=self.training)
            
            # Execute action
            next_state, reward, done = self.step(action)
            total_reward += reward
            
            # Update Q-table if training
            if self.training:
                self.agent.update(state, action, reward, next_state, done)
            
            state = next_state
            
            # Render
            self.draw()
            if self.render_game:
                pygame.display.flip()
                if self.training:
                    self.clock.tick(TRAINING_FPS)  # Fast training
                else:
                    self.clock.tick(FPS)  # Normal speed for testing
            
            if done:
                break
                
        return self.score
        
    def run(self):
        """Main training loop"""
        print("=" * 50)
        print("Flappy Bird Q-Learning Training")
        print("=" * 50)
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
            # Handle events between episodes (important for fast training mode)
            if not self.handle_events():
                break
                
            self.agent.episode += 1
            score = self.run_episode()
            
            if score is None:
                running = False
                break
                
            # Update best score
            if score > self.agent.best_score:
                self.agent.best_score = score
                print(f"New best score: {int(score)} (Episode {self.agent.episode})")
                
            # Decay epsilon after each episode
            if self.training:
                self.agent.decay_epsilon()
                
            # Print progress every 100 episodes
            if self.agent.episode % 100 == 0:
                print(f"Episode {self.agent.episode}: Score={int(score)}, "
                      f"Best={int(self.agent.best_score)}, "
                      f"Epsilon={self.agent.epsilon:.4f}, "
                      f"Q-table size={len(self.agent.q_table)}")
                      
            # Auto-save every 500 episodes
            if self.agent.episode % 500 == 0:
                self.agent.save()
                
        # Save on exit
        self.agent.save()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()

