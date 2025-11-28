import sys
import os

import numpy as np
import matplotlib.pyplot as mplot
from stable_baselines3 import PPO

from bug_platform_env_v2 import BugPlatformEnv
from render_env import BugPlatformEnvWithRender
import pygame


def draw_pygame_background() -> np.ndarray:
    env = BugPlatformEnvWithRender()
    env.reset()
    env.render()
    
    surface = env.screen
    width, height = surface.get_size()
    
    scale = env.scale
    offset = 50
    
    bg = pygame.surfarray.array3d(surface)
    # imshow() hates numpy arrays
    bg = np.transpose(bg, (1, 0, 2))
    bg = np.flipud(bg)
    
    x_min = (0 - offset) / scale
    x_max = (width - offset) / scale
    y_min = (1 - offset) / scale
    y_max = (height - offset) / scale
    
    env.close()
    
    return bg, x_min, x_max, y_min, y_max

def collect_positions(
    model_path: str,
    n_episodes: int = 500,
    max_steps: int | None = None,
    deterministic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    env = BugPlatformEnv()

    xs = []
    ys = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)

            # Gets step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Extract position from internal state: [x, y, vx, vy]
            try:
                x, y, *_ = env.state
            except Exception:
                # In case there is no state
                x, y = float(obs[0]), float(obs[1])
            
            # Center of the player
            y += env.player_height / 2

            xs.append(x)
            ys.append(y)

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        
        # Making sure the program is running and isn't frozen
        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes")

    env.close()
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    print(f"Total collected positions: {xs.shape[0]}")
    return xs, ys


def plot_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    bins_x: int = 300,
    bins_y: int = 120,
    title: str = "State visitation heatmap",
    save_path: str | None = None,
):
    # Debugging
    if xs.size == 0:
        print("No positions to plot.")
        return

    bg, x_min, x_max, y_min, y_max = draw_pygame_background()
    
    heatmap, xedges, yedges = np.histogram2d(
        xs, ys,
        bins = [bins_x, bins_y], 
        range = [[x_min, x_max], [y_min, y_max]],
    )

    mplot.figure(figsize=(15, 6))
    
    # Pygame map background
    mplot.imshow(
        bg,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
    )

    # Heatmap
    mplot.imshow(
        heatmap.T, # imshow() for some reason expects [y, x] here
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
        interpolation="nearest",
        alpha=0.6,
        cmap="hot"
    )
    
    mplot.colorbar(label="Position count")
    mplot.xlabel("x (world units)")
    mplot.ylabel("y (world units)")
    mplot.title(title)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mplot.savefig(save_path, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    mplot.show()


def main():
    # Default model path: final trained model
    model_path = "models/ppo_bug_platform"
    if len(sys.argv) > 1:
        model_path = f"models/checkpoints/{sys.argv[1]}"

    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"Model '{model_path}' not found")
        sys.exit(1)

    # Just in case I forget the .zip
    if os.path.exists(model_path + ".zip"):
        model_path = model_path + ".zip"

    N_EPISODES = 500       # How many episodes to roll out
    MAX_STEPS = None       # In case generating is taking too long or it's useless to see past a certain number of steps
    DETERMINISTIC = False

    xs, ys = collect_positions(
        model_path=model_path,
        n_episodes=N_EPISODES,
        max_steps=MAX_STEPS,
        deterministic=DETERMINISTIC,
    )

    # Plot Resolution;
    BINS_X = 300
    BINS_Y = 120

    base_name = os.path.basename(model_path).replace(".zip", "")
    title = f"Heatmap for {base_name} ({N_EPISODES} episodes)"
    save_path = os.path.join("plots", f"heatmap_{base_name}.png")

    plot_heatmap(
        xs, ys,
        bins_x=BINS_X, bins_y=BINS_Y,
        title=title,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
