import os
import time
import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback


# --- 1. One-Cycle LR Scheduler (picklable class) ---
class GlobalOneCycleLR:
    def __init__(self, max_lr: float, total_steps: int, pct_start: float = 0.1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * pct_start)
        self.initial_lr = max_lr / 25.0
        self.final_lr = max_lr / 10000.0
        self.current_step = 0

    def __call__(self, progress_remaining: float) -> float:
        self.current_step += 1
        t = self.current_step
        if t <= self.warmup_steps:
            return self.initial_lr + (self.max_lr - self.initial_lr) * (t / max(self.warmup_steps, 1))
        else:
            decay_progress = (t - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            return self.max_lr - (self.max_lr - self.final_lr) * min(decay_progress, 1.0)


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

# Enemy-free maps for wall-navigation generalization phase.
# These are used via predefined_map_list so the env cycles through
# them across episodes, giving layout variety without enemy complexity.
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


# --- 3. Curriculum ---
# Order is critical:
#   1. Learn movement on open space first
#   2. Learn walls before ever seeing enemies
#   3. Generalize walls across different layouts (no enemies)
#   4. Introduce enemies in a structured maze
#   5. Generalize to full random maps with enemies
#   6. Specialize on chokepoint for the tournament
curriculum = [
    ("JustGo",          just_go_map,          100_000),
    ("Safe",            safe_map,             200_000),
    ("Random-NoEnemy",  random_no_enemy_maps, 200_000),
    ("Maze",            maze_map,             250_000),
    ("Random-Full",     None,                 300_000),
    ("Chokepoint",      chokepoint_map,       550_000),
]

total_steps = sum(s for _, _, s in curriculum)
print(f"Total training steps: {total_steps:,}")


# --- 4. EvalCallback ---
os.makedirs("./best_model_logs/", exist_ok=True)
eval_env = gym.make("standard", render_mode=None, predefined_map=chokepoint_map)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model_logs/",
    log_path="./best_model_logs/",
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    warn=False
)


# --- 5. Initialize DQN ---
first_env = gym.make("standard", render_mode=None, predefined_map=just_go_map)
lr_schedule = GlobalOneCycleLR(max_lr=5e-4, total_steps=total_steps, pct_start=0.1)

model = DQN(
    "MlpPolicy",
    first_env,
    verbose=1,
    tensorboard_log="./dqn_tensorboard_logs/",
    learning_rate=lr_schedule,
    max_grad_norm=10.0,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=128,
    gamma=0.99,
    exploration_fraction=0.8,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    target_update_interval=1_000,
    train_freq=4,
    gradient_steps=1,
    device="cuda"
)


# --- 6. Curriculum Training Loop ---
for i, (map_name, map_layout, steps) in enumerate(curriculum):
    print(f"\n--- Phase {i+1}/{len(curriculum)}: {map_name} ({steps:,} steps) ---")

    if i > 0:
        if map_layout is None:
            new_env = gym.make("standard", render_mode=None)
        elif isinstance(map_layout, list) and isinstance(map_layout[0][0], list):
            new_env = gym.make("standard", render_mode=None, predefined_map_list=map_layout)
        else:
            new_env = gym.make("standard", render_mode=None, predefined_map=map_layout)
        model.set_env(new_env)

    cb = eval_callback if map_name == "Chokepoint" else None
    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,
        callback=cb,
        tb_log_name="curriculum_run",
    )


# --- 7. Save ---
model.save("group9_dqn_agent")
print("\nFinal model saved as group9_dqn_agent.zip")
print("Best chokepoint model saved in ./best_model_logs/best_model.zip")
print("\nTo view tensorboard: tensorboard --logdir ./dqn_tensorboard_logs/")
input("\nPress Enter to run visual test...")


# --- 8. Visual Test on Chokepoint ---
print("\nRunning visual test on Chokepoint map...")
env_human = gym.make("standard", render_mode="human", predefined_map=chokepoint_map)
obs, _ = env_human.reset()

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, done, trunc, info = env_human.step(action)
    time.sleep(0.1)

    if done or trunc:
        status = "Game Over" if info.get("game_over") else "Cleared or Timeout"
        covered = info.get("total_covered_cells", "?")
        coverable = info.get("coverable_cells", "?")
        print(f"Step {step} | {status} | Coverage: {covered}/{coverable}")
        break

env_human.close()