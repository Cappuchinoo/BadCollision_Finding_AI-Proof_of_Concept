import sys
import time

import numpy as np
from stable_baselines3 import PPO

from bug_platform_env_v2 import BugPlatformEnv
from render_env import BugPlatformEnvWithRender


def run_multi(model_path: str, n_agents: int = 32, deterministic: bool = False):
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    envs = [BugPlatformEnv() for _ in range(n_agents)]
    obs_list = []
    done_list = []
    
    for env in envs:
        obs, info = env.reset()
        obs_list.append(obs)
        done_list.append(False)
    
    renderer = BugPlatformEnvWithRender()
    
    def agent_color(idx: int):
        if idx == 0:
            return (255, 255, 0)
        elif idx < n_agents // 3:
            return (100, 200, 255)
        elif idx < 2 * n_agents // 3:
            return (255, 150, 100)
        else:
            return (180, 255, 180)
    
    try:
        while True:
            positions = []
            
            # Step all agents
            for i, env in enumerate(envs):
                if done_list[i]:
                    obs, info = env.reset()
                    obs_list[i] = obs
                    done_list[i] = False

                action, _ = model.predict(obs_list[i], deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                obs_list[i] = obs
                done_list[i] = terminated or truncated
                
                x, y, vx, vy = env.state
                positions.append((x, y))
            
            # Draw one background frame
            renderer.debug_text = f"Agents: {n_agents} | deterministic={deterministic}"
            renderer.draw_background()
            
            # Draw all agents
            for j, (x, y) in enumerate(positions):
                renderer.draw_player_at(x, y, color=agent_color(j), show_hitbox=False)
            
            renderer.clock.tick(40)
            
            import pygame  # Late import for efficiency
            pygame.display.flip()
    
    except SystemExit:
        pass  # Window is closed
    
    finally:
        for env in envs:
            env.close()
        renderer.close()
    
    
if __name__ == "__main__":
    model_path = "models/checkpoints/ppo_bug_platform_917504_steps.zip"
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith(".zip") or "/" in arg or "\\" in arg:
            model_path = arg
        else:
            model_path = f"models/checkpoints/{arg}"
            if not model_path.endswith(".zip"):
                model_path += ".zip"

    run_multi(model_path=model_path, n_agents=64, deterministic=False)
            