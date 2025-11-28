import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

class BugPlatformEnv(gym.Env):
    """
    Simple 2D platformer with:
    - A square player that can move left/right and jump.
    - A wall with a tiny "buggy" gap where collision fails.
    - A flag to the right (reaching it as fast as possible is the player's objective).
    - Platforms to the left of the wall allow a "legitimate" way to jump -> over <- the wall.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        super().__init__()

        # --- Environment constants ---
        
        # Misc
        self.speed_factor = 0.1  # The higher the value, higher the reward for reaching the flag quickly
        self.y_before_jump = 0.0
        self.is_jumping = False
        self.jump_cooldown = 0
        self.jump_cooldown_max = 40
        self.prev_jump = 0
        
        # Player
        self.player_width = 0.6  # 0.6 because in render_env.py "player_size" is 18 and world scale is 30 (0.6 * 30 = 18)
        self.player_height = 0.6
        self.player_half_width = self.player_width / 2  # Helps so self.player_width / 2 isn't reused everywhere
        self.hitbox_height = self.player_height / 6.0  # The collision is 1/6 of the player
        
        # Physics
        self.dt = 0.02
        self.max_steps = 1000
        self.gravity = -30.0
        self.move_speed = 4.0
        self.jump_speed = 12.0

        # Level layout
        self.start_x = 1.0
        self.start_y = 0.0
        self.flag_x = 20.0
        self.flag_y = 0.0

        # Wall position and fake bug gap
        self.wall_x = 14.0
        self.wall_width = 0.6
        self.wall_height = 8.5
        self.bug_gap_y_min = 5.5   # Only in this vertical band can you "phase" through the wall (as if it's not there)
        self.bug_gap_y_max = self.bug_gap_y_min + (self.hitbox_height * 3)  # "HARD" to find bug
        
        # Section/cycle tracking
        # self.section_width = 5
        # self.current_section = None
        # self.steps_in_section = 0
        # self.max_section_steps = 80   # How long the agent can stay in one section
        
        self.stagnation_window = 100
        self.stagnation_threshhold = 2
        self.recent_x = deque(maxlen=self.stagnation_window)
        
        # Platforms to the left of the wall (x_left, x_right, y_top)
        self.platform_thickness = 0.2
        self.platforms = [
            (9.0, 12.5, 1.5),
            (6.0, 9.5, 3.0),
            (8.5, 12.0, 4.5),
            (5.5, 10.5, 6.0),
            (9.5, 13.5, 7.5),
        ]

        # --- State variables ---
        
        
        # [x, y, vx, vy]
        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, -np.inf, 0.0], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([3, 2])

        self.state = None
        self.steps = 0

        # For rendering (only used in visualize.py)
        self._viewer_initialized = False


    # -----------------------------------------------------------
    # Core RL interface (no Pygame - Better training efficiency)
    # -----------------------------------------------------------
    
    def reset(self, *, seed=None, options=None):
        # Resets the environment before each step
        super().reset(seed=seed)
        
        self.jump_cooldown = 0
        self.is_jumping = False
        self.prev_jump = 0
        
        # self.current_section = int(self.start_x // self.section_width)
        # self.steps_in_section = 0
        
        self.state = np.array([
            self.start_x,
            self.start_y,
            0.0,
            0.0,
            ], dtype=np.float32)
        
        self.steps = 0
        
        self.recent_x.clear()
        self.recent_x.append(self.start_x)
        
        obs = self._get_obs()
        info = {} # PLACEHOLDER (potential future logging)
        return obs, info


    def step(self, action):
        # "Real time" character control and physics logics
        reward = 0.0
        
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        
        x, y, vx, vy = self.state
        self.steps += 1

        movement, jump = action
        
        # To stop jump spamming
        if jump == 1:
            reward -= 0.01
        
        jump_pressed = (jump == 1 and self.prev_jump == 0)
        
        # Horizontal control
        if movement == 1:         # left
            vx = -self.move_speed
        elif movement == 2:       # right
            vx = self.move_speed
        else:
            vx = 0.0

        # Jump: only if "on the ground"
        if jump_pressed and vy < 0.001 and self.jump_cooldown == 0:
            if self._on_ground(x, y):
                self.y_before_jump = y
                self.is_jumping = True
                
                self.jump_cooldown = self.jump_cooldown_max
                
                vy = self.jump_speed
        
        # You have to release jump to be able to jump again
        self.prev_jump = jump

        # Apply gravity
        vy += self.gravity * self.dt

        # Integrate position
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt

        # Ground collision
        if y_new < 0.0:
            y_new = 0.0
            vy = 0.0

        # Platform collision
        if vy < 0.0: # "Falling"
            for (px1, px2, py) in self.platforms:
                if (y >= py + self.platform_thickness) and (y_new <= py + self.platform_thickness):
                    player_left_new = x_new - self.player_half_width
                    player_right_new = x_new + self.player_half_width

                    is_over_platform = (player_right_new > px1) and (player_left_new < px2)
                    
                    if is_over_platform:
                        y_new = py + self.platform_thickness
                        vy = 0.0
                        break

        
        # -- Wall collision with a tiny "bug gap" --
        
        # Numpy coordinates are centered, these are the wall's boundaries
        wall_left = self.wall_x - (self.wall_width / 2)
        wall_right = self.wall_x + (self.wall_width / 2)
        
        # Player collision conditionals, including fake bug handling
        player_left_new = x_new - self.player_half_width
        player_right_new = x_new + self.player_half_width
        hitbox_bottom_new = y_new
        hitbox_top_new = y_new + self.hitbox_height
        
        inside_wall_vertical = (hitbox_bottom_new < self.wall_height) and (hitbox_top_new > 0.0)
        inside_wall_horizontal = (player_right_new > wall_left) and (player_left_new < wall_right)

        in_bug_gap = self.bug_gap_y_min <= y <= self.bug_gap_y_max
        
        # entering_from_left = (x < wall_left) and (x_new >= wall_left)
        # entering_from_right = (x > wall_right) and (x_new <= wall_right)
        within_wall_height = (y < self.wall_height)

        
        if inside_wall_horizontal and inside_wall_vertical:
            if in_bug_gap:
                if hitbox_bottom_new <= self.bug_gap_y_min:
                    y_new = self.bug_gap_y_min
                    vy = 0.0
                elif hitbox_top_new >= self.bug_gap_y_max:
                    y_new = self.bug_gap_y_max - self.hitbox_height

            elif not within_wall_height:
                if hitbox_bottom_new <= self.wall_height:
                    y_new = self.wall_height
                    vy = 0.0
            
            elif x < self.wall_x:
                x_new = wall_left - self.player_half_width
                vx = 0.0
            else:
                x_new = wall_right + self.player_half_width
                vx = 0.0
        
        # if (wall_left <= x_new <= wall_right) and within_wall_height:
        #     if vx > 0:  # Coming from the left
        #         x_new = wall_left
        #     elif vx < 0:  # Coming from the right
        #         x_new = wall_right
        #     else:
        #         if x < self.wall_x:
        #             x_new = wall_left
        #         else:
        #             x_new = wall_right
        #     vx = 0.0
            
            
            # if not (self.bug_gap_y_min <= y_new <= self.bug_gap_y_max):
            #     # Proper collision: stop at wall
            #     x_new = wall_left - 1
            #     vx = 0.0
            # else:
            #     # Bug: allow phasing through (no correction)
            #     pass
        
        # if entering_from_right and within_wall_height:
        #     if not (self.bug_gap_y_min <= y_new <= self.bug_gap_y_max):
        #         # Proper collision: stop at wall
        #         x_new = wall_right
        #         vx = 0.0
        #     else:
        #         # Bug: allow phasing through (no correction)
        #         pass

        self.state = np.array([x_new, y_new, vx, vy], dtype=np.float32)

        # Reward: shaped for reaching flag fast
        eps = 0.001

        # Height and lower distance to the flag are good
        dist_prev = abs(self.flag_x - x)
        dist_curr = abs(self.flag_x - x_new)
        
        reward += (dist_prev - dist_curr) / 5
        if y_new - y > 0:
            reward += (y_new - y) / 10
            # distance_to_wall_abs = abs(self.wall_x - x)
            # if distance_to_wall_abs < 4:
            #    reward += (y_new - y) / 5
            # else:
            #    reward += (y_new - y) / 10
        

        # Step penalty to encourage speed
        reward -= 0.01
        
        # Staying still penalty
        if abs(vx) < eps:
            reward -= 0.05
        
        # Jump penalty/reward
        if self.is_jumping and self._on_ground(x_new, y_new) and abs(vy) < eps:
            height_gained = y_new - self.y_before_jump
            
            # Gained no height after jump
            if height_gained < -0.2:
                pass
            elif height_gained < 0.2:
                reward -= 0.4
            # Gained some height after jump
            else:
                reward += 0.2
            
            self.is_jumping = False
        
        # Cycle/stuck penalty
        self.recent_x.append(x_new)
        
        if len(self.recent_x) == self.stagnation_window:
            x_min = min(self.recent_x)
            x_max = max(self.recent_x)
            if (x_max - x_min) < self.stagnation_threshhold:
                reward -= 0.03
        # section = int(x_new // self.section_width)  # Artificial section (2.1 // 2 == 3.9 // 2)

        # if self.current_section is None:
        #     self.current_section = section
        #     self.steps_in_section = 0
        # elif section == self.current_section:
        #     self.steps_in_section += 1
        # else:
        #     self.current_section = section
        #     self.steps_in_section = 0
        
        # if self.steps_in_section > self.max_section_steps:
        #     time_stuck = self.steps_in_section - self.max_section_steps
        #     penalty = min(0.001 * time_stuck, 0.1)
        #     reward -= penalty
            
            
        # Termination conditions
        terminated = False      # Reached the goal
        truncated = False       # Timed out
        
        # Success condition
        if self._reached_flag(x_new, y_new):
            time_left = self.max_steps - self.steps
            reward += 10.0 + (self.speed_factor * time_left)
            terminated = True

        # Time limit
        if self.steps >= self.max_steps:
            truncated = True

        info = {}
        obs = self._get_obs()
        
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        x, y, vx, vy = self.state
        
        cooldown_norm = self.jump_cooldown / self.jump_cooldown_max
        
        distance_to_flag = self.flag_x - x
        distance_to_wall = self.wall_x - x
        
        grounded = 1.0 if self._on_ground(x, y) else 0.0
        
        return np.array([x, y, vx, vy, cooldown_norm, distance_to_flag, distance_to_wall, grounded], dtype=np.float32)

    # Checks if it's standing on a surface
    def _on_ground(self, x, y, eps = 0.001):
        player_left = x - self.player_half_width
        player_right = x + self.player_half_width
        
        # Ground
        if abs(y) < eps:
            return True

        # Platforms
        for (px1, px2, py) in self.platforms:
            top = py + self.platform_thickness
            if (abs(y - top) < eps):
                is_over_platform = (player_right > px1) and (player_left < px2)
                if is_over_platform:
                    return True
        
        # Wall (and wall "hole")
        if (abs(y - self.wall_height) < eps):
            return True
        elif (abs(y - self.bug_gap_y_min) < eps):
            return True
        
        return False

    def _reached_flag(self, x, y):
        return abs(x - self.flag_x) < 0.5 and abs(y - self.flag_y) < 0.5

    # --------------------------------------------------
    # Rendering: will be implemented with Pygame in visualize.py
    # --------------------------------------------------
    def render(self, mode="human"):
        # Placeholder; in visualize.py you can import Pygame and
        # implement actual drawing using self.state and level layout.
        pass

    def close(self):
        pass
