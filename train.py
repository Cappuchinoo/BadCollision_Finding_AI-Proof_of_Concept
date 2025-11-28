import os
import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from episode_stats_callback import EpisodeStatsCallback

from bug_platform_env_v2 import BugPlatformEnv


def main():
    # Number of parallel environments
    num_cpu = 16   # safer: max(1, mp.cpu_count() // 2)

    env = make_vec_env(
        BugPlatformEnv,   # env class, not an instance
        n_envs=num_cpu,
        seed=0,           # base seed; each env gets a different offset
    )

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,      # per env; total steps per update = n_steps * num_envs
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.08,     # exploration via entropy bonus
        learning_rate=3e-4,
    )

    total_timesteps = 1_000_000
    save_every_timesteps = 131_072
    os.makedirs("models", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq = save_every_timesteps // num_cpu,
        save_path = "models/checkpoints",
        name_prefix = "ppo_bug_platform",
        save_replay_buffer = False,
        save_vecnormalize = False,
    )
    
    stats_callback = EpisodeStatsCallback(verbose=1)

    callback = CallbackList([checkpoint_callback, stats_callback])
    
    model.learn(
        total_timesteps = total_timesteps,
        callback = callback,
    )

    # Save model
    final_path = os.path.join("models", "ppo_bug_platform")
    model.save(final_path)

    env.close()
    print(f"Training finished. Model saved to: {final_path}")


if __name__ == "__main__":
    main()
