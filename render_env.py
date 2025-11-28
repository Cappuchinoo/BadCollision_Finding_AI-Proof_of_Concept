# render_env.py
import pygame
from bug_platform_env_v2 import BugPlatformEnv


class BugPlatformEnvWithRender(BugPlatformEnv):
    def __init__(self):
        super().__init__()

        self.screen = None
        self.clock = None

        self.scale = 30
        self.screen_width = 900
        self.screen_height = 400
        
        self._font = None
        self.debug_text = ""
    
    # Pygame helpers
    
    def _ensure_pygame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Bug Platformer - Rendering")
            self.clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 18)
    
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _world_to_screen(self, wx, wy):
        sx = int(wx * self.scale + 50)
        sy = int(self.screen_height - (wy * self.scale + 50))
        return sx, sy
    
    # Drawing helpers
    
    def draw_background(self):
        self._ensure_pygame()
        self._handle_events()
        
        # Clear background
        self.screen.fill((30, 30, 30))
        
        # ----- Draw ground -----
        gx1, gy1 = self._world_to_screen(-5, 0)
        gx2, gy2 = self._world_to_screen(30, 0)
        pygame.draw.line(self.screen, (200, 200, 200), (gx1, gy1), (gx2, gy2), 2)
        
        # ----- Draw platforms -----
        for (px1, px2, py) in self.platforms:
            sx1, sy1 = self._world_to_screen(px1, py)
            sx2, sy2 = self._world_to_screen(px2, py + 0.2)  # <---- PLATFORM THICKNESS
            rect = pygame.Rect(
                min(sx1, sx2),
                min(sy1, sy2),
                abs(sx2 - sx1),
                abs(sy2 - sy1),
            )
            pygame.draw.rect(self.screen, (150, 150, 150), rect)
        
        # ----- Draw wall -----
        wall_left = self.wall_x - self.wall_width / 2
        wall_right = self.wall_x + self.wall_width / 2
        wall_top = self.wall_height

        x1, y1 = self._world_to_screen(wall_left, 0)
        x2, y2 = self._world_to_screen(wall_right, wall_top)
        wall_rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        pygame.draw.rect(self.screen, (100, 100, 255), wall_rect)
        
        # ----- Draw bug gap region (outline) -----
        gx1, gy1 = self._world_to_screen(
            self.wall_x - self.wall_width / 2, self.bug_gap_y_min
        )
        gx2, gy2 = self._world_to_screen(
            self.wall_x + self.wall_width / 2, self.bug_gap_y_max
        )
        gap_rect = pygame.Rect(min(gx1, gx2), min(gy1, gy2), abs(gx2 - gx1), abs(gy2 - gy1))
        pygame.draw.rect(self.screen, (255, 0, 0), gap_rect, width=1)

        # ----- Draw flag -----
        fx, fy = self._world_to_screen(self.flag_x, self.flag_y)
        flag_rect = pygame.Rect(fx - 5, fy - 40, 10, 40)
        pygame.draw.rect(self.screen, (0, 255, 0), flag_rect)
        
        if self.debug_text:
            text_surface = self._font.render(self.debug_text, True, (255, 255, 200))
            self.screen.blit(text_surface, (10, 10))
    
    def draw_player_at(self, x, y, color=(255, 255, 0), show_hitbox=True):
        px, py = self._world_to_screen(x, y)
        player_size = int(self.player_width * self.scale)
        collision_height = player_size / 6

        player_rect = pygame.Rect(px - player_size // 2, py - player_size, player_size, player_size)
        pygame.draw.rect(self.screen, color, player_rect)
        
        if show_hitbox:
            collision_rect = pygame.Rect(player_rect.left, player_rect.bottom - collision_height, player_size, collision_height)
            pygame.draw.rect(self.screen, (255, 0, 50), collision_rect)


    # Single agent render
    def render(self):
        self.draw_background()
        
        x, y, vx, vy = self.state
        self.draw_player_at(x, y, color=(255, 255, 0), show_hitbox=True)
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
        super().close()

