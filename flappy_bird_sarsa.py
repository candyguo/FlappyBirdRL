"""
Flappy Bird with SARSA (State-Action-Reward-State-Action)

SARSA is an ON-POLICY TD control algorithm.
Key difference from Q-Learning:
  - Q-Learning (off-policy): Q(s,a) += α * [r + γ * max_a' Q(s',a') - Q(s,a)]
  - SARSA      (on-policy):  Q(s,a) += α * [r + γ * Q(s',a')         - Q(s,a)]

SARSA uses the action ACTUALLY TAKEN in the next state (a'), not the best possible.
This makes it more conservative — it accounts for exploration mistakes in its value estimates.

Name origin: the quintuple (S, A, R, S', A') needed for each update.
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
TRAINING_FPS = 240

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

# SARSA parameters (same as Q-Learning for fair comparison)
LEARNING_RATE = 0.1      # α - learning rate
DISCOUNT_FACTOR = 0.95   # γ - discount factor
EPSILON_START = 1.0      # Initial exploration rate
EPSILON_MIN = 0.01       # Minimum exploration rate
EPSILON_DECAY = 0.9995   # Decay rate per episode

# Rewards
REWARD_ALIVE = 1
REWARD_DEAD = -1000

# State discretization bins
X_BINS = 20
Y_BINS = 20
VY_BINS = 10


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


class SARSAAgent:
    """
    SARSA Agent — On-policy TD Control

    The critical difference from Q-Learning is in the update() method:
    - Q-Learning uses: max_a' Q(s', a')     ← best possible next action
    - SARSA uses:      Q(s', a')            ← actual next action (chosen by ε-greedy)

    This means SARSA's Q-values reflect the ACTUAL policy being followed
    (including random exploration), not the theoretical optimal policy.
    """
    def __init__(self):
        self.q_table = defaultdict(float)
        self.epsilon = EPSILON_START
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR

        self.episode = 0
        self.best_score = 0
        self.scores_history = []

    def discretize_state(self, dx, dy, velocity_y):
        """Discretize continuous state into discrete bins"""
        dx_bin = int(max(0, min(X_BINS - 1, dx * X_BINS / PIPE_SPACING)))

        dy_normalized = (dy + SCREEN_HEIGHT) / (2 * SCREEN_HEIGHT)
        dy_bin = int(max(0, min(Y_BINS - 1, dy_normalized * Y_BINS)))

        vy_normalized = (velocity_y + 15) / 30
        vy_bin = int(max(0, min(VY_BINS - 1, vy_normalized * VY_BINS)))

        return (dx_bin, dy_bin, vy_bin)

    def get_action(self, state, training=True):
        """ε-greedy action selection: 0 (no jump) or 1 (jump)"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            q_no_jump = self.q_table[(state, 0)]
            q_jump = self.q_table[(state, 1)]

            if q_jump > q_no_jump:
                return 1
            elif q_no_jump > q_jump:
                return 0
            else:
                return random.randint(0, 1)

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update rule:
          Q(s,a) ← Q(s,a) + α * [ r + γ * Q(s',a') - Q(s,a) ]

        Compare with Q-Learning:
          Q(s,a) ← Q(s,a) + α * [ r + γ * max_a' Q(s',a') - Q(s,a) ]
                                            ^^^^^
                                            This is the ONLY difference!

        Parameters:
            state:       current state s
            action:      action taken a
            reward:      reward received r
            next_state:  next state s'
            next_action: action ACTUALLY chosen for s' (the second A in SARSA)
            done:        whether episode ended
        """
        current_q = self.q_table[(state, action)]

        if done:
            # Terminal state: no future reward
            next_q = 0
        else:
            # ★ SARSA: use Q-value of the action we will ACTUALLY take
            # Q-Learning would do: max(Q(s',0), Q(s',1))
            next_q = self.q_table[(next_state, next_action)]

        # TD update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save(self, filename="sarsa_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'episode': self.episode,
                'best_score': self.best_score
            }, f)
        print(f"Model saved to {filename}")

    def load(self, filename="sarsa_table.pkl"):
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
            pygame.display.set_caption("Flappy Bird - SARSA")
            self.clock = pygame.time.Clock()
            self.font_large = pygame.font.Font(None, 72)
            self.font_medium = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 28)

        self.agent = SARSAAgent()
        self.training = False
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
        dx = pipe_x - self.bird.x
        dy = self.bird.y - pipe_y

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
        info_texts = [
            f"Algorithm: SARSA (On-Policy)",
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
        """Execute one game step. Returns: (next_state, reward, done)"""
        if action == 1:
            self.bird.jump()

        self.bird.update()

        for pipe in self.pipes:
            pipe.update(BIRD_SPEED_X)

        if self.pipes and self.pipes[0].is_offscreen():
            self.pipes.pop(0)
            new_x = self.pipes[-1].x + PIPE_SPACING
            self.pipes.append(Pipe(new_x))

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
                    self.agent = SARSAAgent()
                    print("Agent reset!")
        return True

    def run_episode(self):
        """
        Run one SARSA episode.

        ★ KEY DIFFERENCE FROM Q-LEARNING IN THE TRAINING LOOP:

        Q-Learning loop:
            state = get_state()
            while not done:
                action = choose(state)          # choose action
                next_state, reward, done = step(action)
                update(state, action, reward, next_state)  # uses max Q(s',·)
                state = next_state

        SARSA loop:
            state = get_state()
            action = choose(state)              # ★ choose FIRST action upfront
            while not done:
                next_state, reward, done = step(action)
                next_action = choose(next_state)  # ★ choose NEXT action BEFORE update
                update(state, action, reward, next_state, next_action)  # ★ pass next_action
                state = next_state
                action = next_action              # ★ carry forward
        """
        self.reset_game()
        state = self.get_state()

        # ★ SARSA: choose the first action BEFORE entering the loop
        action = self.agent.get_action(state, training=self.training)

        while True:
            if not self.handle_events():
                return None

            # Execute the pre-chosen action
            next_state, reward, done = self.step(action)

            # ★ SARSA: choose next action NOW (before update)
            # This next_action is what we'll ACTUALLY do next,
            # and also what we use in the Q-value update
            next_action = self.agent.get_action(next_state, training=self.training)

            # ★ SARSA update: pass next_action explicitly
            if self.training:
                self.agent.update(state, action, reward, next_state, next_action, done)

            # Transition: carry forward state AND action
            state = next_state
            action = next_action

            self.draw()
            if self.render_game:
                pygame.display.flip()
                if self.training:
                    self.clock.tick(TRAINING_FPS)
                else:
                    self.clock.tick(FPS)

            if done:
                break

        return self.score

    def run(self):
        print("=" * 50)
        print("Flappy Bird SARSA Training")
        print("=" * 50)
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
            score = self.run_episode()

            if score is None:
                running = False
                break

            if score > self.agent.best_score:
                self.agent.best_score = score
                print(f"New best score: {int(score)} (Episode {self.agent.episode})")

            if self.training:
                self.agent.decay_epsilon()

            if self.agent.episode % 100 == 0:
                print(f"Episode {self.agent.episode}: Score={int(score)}, "
                      f"Best={int(self.agent.best_score)}, "
                      f"Epsilon={self.agent.epsilon:.4f}, "
                      f"Q-table size={len(self.agent.q_table)}")

            if self.agent.episode % 500 == 0:
                self.agent.save()

        self.agent.save()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(render=True)
    game.run()
