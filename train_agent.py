import os
import time
import numpy as np
import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import warnings

# --- SMOKE TEST FLAG ---
# True  = 5k steps per phase (~2 mins, verifies no crashes)
# False = full training run
SMOKE_TEST = False

N_ENVS = 8


# --- 1. Environment factory ---
# No wrapper needed — observation and reward are fully self-contained in custom.py.
# The model zip + custom.py is all the tournament needs.
def make_env(map_layout=None, map_list=None):
    def _init():
        if map_list is not None:
            return gym.make("standard", render_mode=None, predefined_map_list=map_list)
        elif map_layout is not None:
            return gym.make("standard", render_mode=None, predefined_map=map_layout)
        else:
            return gym.make("standard", render_mode=None)
    return _init


# --- 2. Maps ---
just_go_map = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

safe_map = [
    [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
]

maze_map = [
    [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
]

chokepoint_map = [
    [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
]

random_no_enemy_maps = [
    [
        [3, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 2, 0, 2, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 2, 2, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [2, 0, 2, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 2, 0, 2, 2, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
    ],
    [
        [3, 0, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 2, 2, 0, 2, 0, 2, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 2, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 2, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 2, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0]
    ],
]

curriculum = [
    ("JustGo",     just_go_map,     100_000),
    ("Safe",       safe_map,        200_000),
    ("Maze",       maze_map,        300_000),
    ("Chokepoint", chokepoint_map,  300_000),
    ("JustGo",     just_go_map,     50_000),
    ("Safe",       safe_map,        50_000),
    ("Maze",       maze_map,        50_000),
    ("Chokepoint", chokepoint_map,  50_000),
    ("JustGo",     just_go_map,     50_000),
    ("Safe",       safe_map,        50_000),
    ("Maze",       maze_map,        50_000),
    ("Chokepoint", chokepoint_map,  100_000),
]


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="Training and eval env are not of the same type"
    )

    total_steps = sum(s for _, _, s in curriculum)
    print(f"Total training steps: {total_steps:,} ({'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'})")
    print(f"Parallel envs: {N_ENVS}")

    # --- EvalCallback ---
    os.makedirs("./best_model_logs/", exist_ok=True)
    eval_env = make_vec_env(make_env(map_layout=chokepoint_map), n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_logs/",
        log_path="./best_model_logs/",
        eval_freq=max(5_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        warn=False
    )

    # --- Initialize PPO ---
    first_vec_env = SubprocVecEnv(
        [make_env(map_layout=just_go_map) for _ in range(N_ENVS)]
    )

    model = PPO(
        "MlpPolicy",
        first_vec_env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=512,
        batch_size=1024,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.04,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # --- Curriculum Training Loop ---
    for i, (map_name, map_layout, steps) in enumerate(curriculum):
        print(f"\n--- Phase {i+1}/{len(curriculum)}: {map_name} ({steps:,} steps) ---")

        if i > 0:
            if map_layout is None:
                new_vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
            elif isinstance(map_layout, list) and isinstance(map_layout[0][0], list):
                new_vec_env = SubprocVecEnv([make_env(map_list=map_layout) for _ in range(N_ENVS)])
            else:
                new_vec_env = SubprocVecEnv([make_env(map_layout=map_layout) for _ in range(N_ENVS)])
            model.set_env(new_vec_env)

        cb = eval_callback if map_name == "Chokepoint" else None
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=cb,
            tb_log_name="ppo_curriculum_run",
        )

    print("\n=== Training Complete ===")

    # --- Save ---
    model.save("group9_ppo_agent")
    print("Final model saved as group9_ppo_agent.zip")
    print("Best chokepoint model saved in ./best_model_logs/best_model.zip")
    print("\nTo view tensorboard: tensorboard --logdir ./ppo_tensorboard_logs/")

    if not SMOKE_TEST:
        input("\nPress Enter to run visual test...")

        print("\nRunning visual test on Chokepoint map...")
        env_human = gym.make("standard", render_mode="human", predefined_map=chokepoint_map)
        obs, _ = env_human.reset()

        for step in range(500):
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            obs, rew, done, trunc, info = env_human.step(int(action[0]))
            time.sleep(0.1)

            if done or trunc:
                status = "Game Over" if info.get("game_over") else "Cleared or Timeout"
                covered = info.get("total_covered_cells", "?")
                coverable = info.get("coverable_cells", "?")
                print(f"Step {step} | {status} | Coverage: {covered}/{coverable}")
                break

        env_human.close()
    else:
        print("\nSmoke test complete — set SMOKE_TEST = False for full training run.")