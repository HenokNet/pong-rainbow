import pygame
import numpy as np
import random
from collections import deque
from .config import *
from .state import StateEncoder
from agent.rainbow import RainbowDQNAgent
from utils.logger import PerformanceLogger

class PongGame:
    def __init__(self, headless=False):
        self.headless = headless
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.font = pygame.font.Font(None, 36)
        
        self.clock = pygame.time.Clock()
        self.ball = pygame.Rect(WIDTH//2-10, HEIGHT//2-10, 20, 20)
        self.player = pygame.Rect(50, HEIGHT//2-50, 10, 100)
        self.ai = pygame.Rect(WIDTH-60, HEIGHT//2-50, 10, 100)
        
        self.reset()
        self.episode = 0
        self.best_rally = 0
        self.total_reward = 0
        
        self.agent = RainbowDQNAgent()
        self.state_encoder = StateEncoder(WIDTH, HEIGHT)
        self.logger = PerformanceLogger()
        self.last_state = None
        self.last_action = None

    def reset(self):
        self.ball.center = (WIDTH//2, HEIGHT//2)
        self.ball_vx = BALL_SPEED * random.choice([1, -1])
        self.ball_vy = BALL_SPEED * random.uniform(-0.5, 0.5)
        self.rally = 0
        self.ai_speed = 0
        self.player_speed = 0
        self.total_reward = 0

    def run(self, max_episodes=0):
        running = True
        episode_count = 0
        
        try:
            while running and (max_episodes <= 0 or episode_count < max_episodes):
                if not self.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                
                state = self.state_encoder.get_state(
                    self.ball, self.ai, self.ball_vx, self.ball_vy)
                
                action = self.agent.act(state, training=True)
                reward = self._calculate_reward(False)
                self.total_reward += reward
                
                self._update_physics(action)
                done = self._check_collisions()
                
                if not self.headless:
                    self._render()
                
                if done:
                    episode_count += 1
                    self._log_episode()
                    if self.episode % 100 == 0:
                        self._save_progress()
                    self.reset()
                
                self.clock.tick(FPS if not self.headless else 1000)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted - saving final results...")
        finally:
            self._save_progress()
            if not self.headless:
                pygame.quit()

    def _log_episode(self):
        self.episode += 1
        self.logger.log_episode(
            reward=self.total_reward,
            length=self.rally,
            rally=self.rally,
            loss=self.agent.last_loss,
            epsilon=self.agent.epsilon
        )
        
        if self.episode % 10 == 0:
            stats = self.logger.get_running_stats()
            print(f"Ep {self.episode:4d} | "
                  f"Reward {stats['reward']:6.1f} | "
                  f"Rally {self.rally:3d} (Best {self.best_rally:3d}) | "
                  f"ε {self.agent.epsilon:.3f} | "
                  f"Loss {self.agent.last_loss:.4f}")

    def _save_progress(self):
        self.logger.visualize_progress()
        self.logger.save_training_report()
        self.best_rally = max(self.best_rally, self.rally)

    def _update_physics(self, action):
        prev_y = self.ai.centery
        if action == 0 and self.ai.top > 0:
            self.ai.y -= PADDLE_SPEED
        elif action == 2 and self.ai.bottom < HEIGHT:
            self.ai.y += PADDLE_SPEED
        self.ai_speed = self.ai.centery - prev_y
        
        if self.ball_vx < 0:
            target_y = self.ball.centery + random.uniform(-20, 20)
            if target_y < self.player.centery and self.player.top > 0:
                self.player.y -= PADDLE_SPEED * 0.9
            elif target_y > self.player.centery and self.player.bottom < HEIGHT:
                self.player.y += PADDLE_SPEED * 0.9
        
        self.ball.x += self.ball_vx
        self.ball.y += self.ball_vy

    def _check_collisions(self):
        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.ball_vy *= -1
            
        if self.ball.colliderect(self.player) or self.ball.colliderect(self.ai):
            paddle = self.player if self.ball.colliderect(self.player) else self.ai
            paddle_speed = self.player_speed if paddle == self.player else self.ai_speed
            
            relative_intersect = (paddle.centery - self.ball.centery) / (paddle.height/2)
            bounce_angle = relative_intersect * (5*np.pi/12)
            speed_influence = paddle_speed * 0.2
            bounce_angle = np.clip(bounce_angle + speed_influence, -5*np.pi/12, 5*np.pi/12)
            
            self.ball_vx = -self.ball_vx * 1.02
            self.ball_vy = -BALL_SPEED * np.sin(bounce_angle)
            self.rally += 1
            self.best_rally = max(self.best_rally, self.rally)
        
        if self.ball.left <= 0 or self.ball.right >= WIDTH:
            return True
        return False

    def _render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.player)
        pygame.draw.rect(self.screen, (255, 255, 255), self.ai)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        pygame.draw.aaline(self.screen, (100, 100, 100), 
                         (WIDTH//2, 0), (WIDTH//2, HEIGHT))
        
        info = [
            f"Episode: {self.episode}",
            f"Rally: {self.rally} (Best: {self.best_rally})",
            f"ε: {self.agent.epsilon:.3f}",
            f"Total Reward: {self.total_reward:.1f}"
        ]
        
        for i, text in enumerate(info):
            surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surf, (20, 20 + i * 30))
        
        pygame.display.flip()

    def _calculate_reward(self, done):
        if done:
            if self.ball.left <= 0:
                return 5.0 + 0.1 * self.rally
            else:
                return -1.0 - 0.05 * self.rally
        
        pred_y = self.ball.y + self.ball_vy * 20
        distance_reward = 1.0 - min(abs(self.ai.centery - pred_y)/HEIGHT, 1.0)
        velocity_match = 1.0 - min(abs(self.ball_vy - self.ai_speed)/10.0, 1.0)
        rally_bonus = 0.01 * self.rally
        direction_bonus = 0.5 if self.ball_vx > 0 else -0.2
        
        return (2.0 * distance_reward + 
                1.0 * velocity_match + 
                rally_bonus + 
                direction_bonus - 
                0.01 * (abs(self.ai_speed) > 0))