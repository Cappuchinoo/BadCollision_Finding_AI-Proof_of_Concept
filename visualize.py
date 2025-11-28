import time
import sys

from stable_baselines3 import PPO

from render_env import BugPlatformEnvWithRender


def play_episodes(model_path: str, n_episodes: int = 1, deterministic: bool = True):
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    env = BugPlatformEnvWithRender()

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # Model expects obs shape (n_envs, obs_dim) but VecEnv is not used here,
            # so we let SB3 handle the single-env case.
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            steps += 1
            
            # Debug - show reward on replays in real time
            env.debug_text = f"Ep reward: {ep_reward: .2f} | Steps: {steps}"

            env.render()

        print(f"Episode {ep}: steps={steps}, total_reward={ep_reward:.2f}")
        # Short pause between episodes
        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    model_path = "models/ppo_bug_platform"
    if len(sys.argv) > 1:
        model_path = f"models/checkpoints/{sys.argv[1]}"
    
    play_episodes(model_path, n_episodes=50, deterministic=True)
