"""
Flappy Bird Game
A simple implementation using pygame
"""

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 640
FPS = 60

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
GAME_OVER_BG = (20, 20, 30, 200)

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


class Bird:
    def __init__(self):
        self.x = 80
        self.y = SCREEN_HEIGHT // 2
        self.velocity_y = 0
        self.radius = 20
        self.angle = 0
        self.flap_offset = 0
        self.flap_timer = 0
        
    def jump(self):
        self.velocity_y = JUMP_STRENGTH
        self.flap_timer = 10
        
    def update(self):
        # Apply gravity
        self.velocity_y += GRAVITY
        self.y += self.velocity_y
        
        # Update angle based on velocity
        self.angle = max(-30, min(60, self.velocity_y * 3))
        
        # Wing flap animation
        if self.flap_timer > 0:
            self.flap_timer -= 1
            self.flap_offset = 8 * (self.flap_timer / 10)
        else:
            self.flap_offset = 0
        
    def draw(self, screen):
        # Draw bird body (circle)
        pygame.draw.circle(screen, BIRD_COLOR, (int(self.x), int(self.y)), self.radius)
        
        # Draw wing
        wing_y = self.y + self.flap_offset - 5
        wing_points = [
            (self.x - 5, wing_y),
            (self.x - 20, wing_y + 5),
            (self.x - 5, wing_y + 10)
        ]
        pygame.draw.polygon(screen, (255, 170, 0), wing_points)
        
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
        
        # Main body
        pygame.draw.rect(screen, PIPE_COLOR, 
                        (self.x, 0, PIPE_WIDTH, top_height))
        # Highlight
        pygame.draw.rect(screen, PIPE_HIGHLIGHT, 
                        (self.x, 0, 8, top_height))
        # Shadow
        pygame.draw.rect(screen, PIPE_SHADOW, 
                        (self.x + PIPE_WIDTH - 8, 0, 8, top_height))
        # Cap
        pygame.draw.rect(screen, PIPE_COLOR, 
                        (self.x - 5, top_height - 30, PIPE_WIDTH + 10, 30))
        pygame.draw.rect(screen, PIPE_HIGHLIGHT, 
                        (self.x - 5, top_height - 30, 8, 30))
        
        # Bottom pipe
        bottom_y = self.gap_y + PIPE_GAP // 2
        bottom_height = SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y
        
        # Main body
        pygame.draw.rect(screen, PIPE_COLOR, 
                        (self.x, bottom_y, PIPE_WIDTH, bottom_height))
        # Highlight
        pygame.draw.rect(screen, PIPE_HIGHLIGHT, 
                        (self.x, bottom_y, 8, bottom_height))
        # Shadow
        pygame.draw.rect(screen, PIPE_SHADOW, 
                        (self.x + PIPE_WIDTH - 8, bottom_y, 8, bottom_height))
        # Cap
        pygame.draw.rect(screen, PIPE_COLOR, 
                        (self.x - 5, bottom_y, PIPE_WIDTH + 10, 30))
        pygame.draw.rect(screen, PIPE_HIGHLIGHT, 
                        (self.x - 5, bottom_y, 8, 30))
        
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
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        self.reset_game()
        
    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.score_timer = 0
        self.game_over = False
        self.started = False
        self.camera_x = 0
        self.ground_offset = 0
        
        # Generate initial pipes
        for i in range(4):
            self.pipes.append(Pipe(SCREEN_WIDTH + i * PIPE_SPACING))
            
    def draw_gradient_background(self):
        for y in range(SCREEN_HEIGHT - GROUND_HEIGHT):
            ratio = y / (SCREEN_HEIGHT - GROUND_HEIGHT)
            r = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * ratio)
            g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * ratio)
            b = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))
            
    def draw_ground(self):
        # Ground top line
        pygame.draw.rect(self.screen, GROUND_TOP, 
                        (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, 10))
        # Ground body
        pygame.draw.rect(self.screen, GROUND_COLOR, 
                        (0, SCREEN_HEIGHT - GROUND_HEIGHT + 10, SCREEN_WIDTH, GROUND_HEIGHT - 10))
        
        # Ground pattern
        pattern_width = 40
        self.ground_offset = (self.ground_offset + BIRD_SPEED_X) % pattern_width
        for x in range(-pattern_width + int(self.ground_offset), SCREEN_WIDTH + pattern_width, pattern_width):
            pygame.draw.line(self.screen, GROUND_TOP, 
                           (x, SCREEN_HEIGHT - GROUND_HEIGHT + 10),
                           (x + 20, SCREEN_HEIGHT), 3)
            
    def draw_score(self):
        # Draw current score with shadow
        score_text = str(int(self.score))
        shadow = self.font_large.render(score_text, True, BLACK)
        text = self.font_large.render(score_text, True, SCORE_COLOR)
        
        x = SCREEN_WIDTH // 2 - text.get_width() // 2
        y = 50
        
        self.screen.blit(shadow, (x + 3, y + 3))
        self.screen.blit(text, (x, y))
        
    def draw_start_screen(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font_large.render("FLAPPY BIRD", True, BIRD_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 150))
        
        # Instructions
        inst1 = self.font_small.render("Press SPACE to flap", True, WHITE)
        inst2 = self.font_small.render("Avoid the pipes!", True, WHITE)
        inst3 = self.font_medium.render("Press SPACE to start", True, (150, 255, 150))
        
        self.screen.blit(inst1, (SCREEN_WIDTH // 2 - inst1.get_width() // 2, 280))
        self.screen.blit(inst2, (SCREEN_WIDTH // 2 - inst2.get_width() // 2, 320))
        self.screen.blit(inst3, (SCREEN_WIDTH // 2 - inst3.get_width() // 2, 400))
        
    def draw_game_over(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        go_text = self.font_large.render("GAME OVER", True, (255, 80, 80))
        self.screen.blit(go_text, (SCREEN_WIDTH // 2 - go_text.get_width() // 2, 180))
        
        # Final score
        score_label = self.font_medium.render("Score", True, WHITE)
        score_value = self.font_large.render(str(int(self.score)), True, BIRD_COLOR)
        
        self.screen.blit(score_label, (SCREEN_WIDTH // 2 - score_label.get_width() // 2, 280))
        self.screen.blit(score_value, (SCREEN_WIDTH // 2 - score_value.get_width() // 2, 330))
        
        # Restart instruction
        restart = self.font_small.render("Press SPACE to restart", True, (150, 255, 150))
        self.screen.blit(restart, (SCREEN_WIDTH // 2 - restart.get_width() // 2, 450))
        
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
        
    def update(self):
        if not self.started or self.game_over:
            return
            
        # Update bird
        self.bird.update()
        
        # Update pipes
        for pipe in self.pipes:
            pipe.update(BIRD_SPEED_X)
            
        # Remove offscreen pipes and add new ones
        if self.pipes and self.pipes[0].is_offscreen():
            self.pipes.pop(0)
            new_x = self.pipes[-1].x + PIPE_SPACING
            self.pipes.append(Pipe(new_x))
            
        # Check collision
        if self.check_collision():
            self.game_over = True
            return
            
        # Update score (every 0.1 second = every 6 frames at 60 FPS)
        self.score_timer += 1
        if self.score_timer >= 6:
            self.score += 1
            self.score_timer = 0
            
    def draw(self):
        # Draw background
        self.draw_gradient_background()
        
        # Draw pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)
            
        # Draw ground
        self.draw_ground()
        
        # Draw bird
        self.bird.draw(self.screen)
        
        # Draw score
        self.draw_score()
        
        # Draw start screen or game over
        if not self.started:
            self.draw_start_screen()
        elif self.game_over:
            self.draw_game_over()
            
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if not self.started:
                            self.started = True
                            self.bird.jump()
                        elif self.game_over:
                            self.reset_game()
                            self.started = True
                            self.bird.jump()
                        else:
                            self.bird.jump()
                            
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
            self.update()
            self.draw()
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()

