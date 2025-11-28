import pygame
import numpy as np
from render_env import BugPlatformEnvWithRender

def play_manual():
    pygame.init()
    pygame.display.set_mode((1, 1))
    
    env = BugPlatformEnvWithRender()
    obs, info = env.reset()

    ep_reward = 0.0
    ep_steps = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
    
        keys = pygame.key.get_pressed()

        act = np.array([0, 0], dtype=int)
        if keys[pygame.K_LEFT]:  act[0] = 1
        if keys[pygame.K_RIGHT]: act[0] = 2
        if keys[pygame.K_SPACE]: act[1] = 1

        obs, reward, terminated, truncated, info = env.step(act)
        
        ep_reward += reward
        ep_steps += 1
        
        env.debug_text = f"Manual | Ep reward: {ep_reward: .2f} | Steps: {ep_steps}"
        
        env.render()

        if terminated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            ep_reward = 0.0
            ep_steps = 0

if __name__ == "__main__":
    play_manual()
