import os
import time
import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# --- 1. Configuration & Reward Logic ---
REWARD_MODE = "balanced"  # Change to "cautious" or "speed" as needed
STAY_ACTION_INDEX = None  # Update this if your agent has a specific integer for "stay still"

def calculate_custom_reward(info: dict) -> float:
    """Updated reward function that safely isolates wall hits from backtracking."""
    if REWARD_MODE == "balanced":
        r = -0.3  # Base step penalty on all actions including STAY

        if info["new_cell_covered"]:
            cells_done = info["coverable_cells"] - info["cells_remaining"]
            progress = cells_done / max(info["coverable_cells"], 1)
            r += 10.0 + (15.0 * progress)  # +10 early, up to +25 late
            
        elif info.get("wall_hit", False):
            # Heavy penalty strictly for hitting a wall, NOT backtracking
            r -= 1.5

        if info["game_over"]:
            # Scale death penalty by cells remaining
            r -= 50.0 + (info["cells_remaining"] * 1.5)

        if info["cells_remaining"] == 0:
            r += 1000.0 + (info["steps_remaining"] * 1.0)

    elif REWARD_MODE == "cautious":
        r = -0.3
        if info["new_cell_covered"]:
            r += 8.0
        elif info.get("wall_hit", False):
            r -= 1.0  # Wall hit penalty
        if info["game_over"]:
            r -= 150.0

    else:  # speed
        r = -1.0
        if info["new_cell_covered"]:
            r += 5.0
        if info["cells_remaining"] == 0:
            r += 300.0

    return float(r)


# --- 2. Environment Wrapper ---
class WallDetectionWrapper(gym.Wrapper):
    """
    Tracks agent position to detect wall collisions and recalculates 
    the reward using our custom logic before giving it to SB3.
    """
    def __init__(self, env, stay_action_index=None):
        super().__init__(env)
        self.previous_pos = None
        self.stay_action_index = stay_action_index

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_pos = info.get("agent_pos")
        return obs, info

    def step(self, action):
        obs, base_reward, done, trunc, info = self.env.step(action)
        
        current_pos = info.get("agent_pos")
        hit_wall = (current_pos == self.previous_pos) and (action != self.stay_action_index)
        info["wall_hit"] = hit_wall
        self.previous_pos = current_pos
        
        # Override the base environment reward with our custom reward
        new_reward = calculate_custom_reward(info)
        
        return obs, new_reward, done, trunc, info


# --- 3. One-Cycle LR Scheduler (picklable class) ---
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


# --- 4. Maps ---
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
]

curriculum = [
    ("JustGo",          just_go_map,          150_000), 
    ("Safe",            safe_map,             250_000), 
    ("Random-NoEnemy",  random_no_enemy_maps, 250_000), 
    ("Maze",            maze_map,             300_000), 
    ("Random-Full",     None,                 350_000), 
    ("Chokepoint",      chokepoint_map,       600_000), 
]

total_steps = sum(s for _, _, s in curriculum)
print(f"Total training steps: {total_steps:,}")


# --- 5. EvalCallback ---
os.makedirs("./best_model_logs/", exist_ok=True)
base_eval_env = gym.make("standard", render_mode=None, predefined_map=chokepoint_map)
eval_env = WallDetectionWrapper(base_eval_env, stay_action_index=STAY_ACTION_INDEX)

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


# --- 6. Initialize DQN ---
base_first_env = gym.make("standard", render_mode=None, predefined_map=just_go_map)
first_env = WallDetectionWrapper(base_first_env, stay_action_index=STAY_ACTION_INDEX)

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


# --- 7. Curriculum Training Loop ---
for i, (map_name, map_layout, steps) in enumerate(curriculum):
    print(f"\n--- Phase {i+1}/{len(curriculum)}: {map_name} ({steps:,} steps) ---")

    if i > 0:
        if map_layout is None:
            base_new_env = gym.make("standard", render_mode=None)
        elif isinstance(map_layout, list) and isinstance(map_layout[0][0], list):
            base_new_env = gym.make("standard", render_mode=None, predefined_map_list=map_layout)
        else:
            base_new_env = gym.make("standard", render_mode=None, predefined_map=map_layout)
            
        new_env = WallDetectionWrapper(base_new_env, stay_action_index=STAY_ACTION_INDEX)
        model.set_env(new_env)

    cb = eval_callback if map_name == "Chokepoint" else None
    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,
        callback=cb,
        tb_log_name="curriculum_run",
    )


# --- 8. Save ---
model.save("group9_dqn_agent")
print("\nFinal model saved as group9_dqn_agent.zip")
print("Best chokepoint model saved in ./best_model_logs/best_model.zip")
print("\nTo view tensorboard: tensorboard --logdir ./dqn_tensorboard_logs/")
input("\nPress Enter to run visual test...")


# --- 9. Visual Test ---
print("\nRunning visual test on Chokepoint map...")
base_env_human = gym.make("standard", render_mode="human", predefined_map=chokepoint_map)
env_human = WallDetectionWrapper(base_env_human, stay_action_index=STAY_ACTION_INDEX)
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