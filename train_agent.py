import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import time

CPUS = 8

# 1. Curriculum Maps (ordered easy → hard)
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

# ... [KEEP YOUR CURRICULUM MAPS LIST HERE] ...
curriculum = [
    ("JustGo",     just_go_map,     150_000),
    ("Safe",       safe_map,        150_000),
    ("Maze",       maze_map,        150_000),
    ("Chokepoint", chokepoint_map,  150_000),
    ("JustGo",     just_go_map,     50_000),
    ("Safe",       safe_map,        50_000),
    ("Maze",       maze_map,        50_000),
    ("Chokepoint", chokepoint_map,  50_000),
    ("JustGo",     just_go_map,     50_000),
    ("Safe",       safe_map,        50_000),
    ("Maze",       maze_map,        50_000),
    ("Chokepoint", chokepoint_map,  100_000),
]
total_training = sum(s for _, _, s in curriculum)
print(f"Total timesteps: {total_training}")

# 1. Setup Checkpoint Saving (Saves every 50k steps per environment | CPUS=4)
# Note: In a VecEnv with 4 envs, save_freq=12500 means it saves every 50,000 total steps
checkpoint_callback = CheckpointCallback(
    save_freq=12500, 
    save_path='./ppo_checkpoints/',
    name_prefix='ppo_agent'
)

# 2. Create Parallel Environments
# n_envs=4 means it runs 4 games simultaneously. Change to 8 if your CPU has 8+ cores!
env_kwargs = {"render_mode": None, "predefined_map": curriculum[0][1]}
first_vec_env = make_vec_env("standard", n_envs=CPUS, env_kwargs=env_kwargs)

# 3. Initialize PPO
model = PPO(
    "MlpPolicy",
    first_vec_env,
    tensorboard_log="./ppo_tensorboard_logs/",
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.94,
    ent_coef=0.06,  
    policy_kwargs=dict(net_arch=[256, 128, 64]) # they call me "the funnel"
)


# 4. Train across the curriculum
for i, (map_name, map_layout, steps) in enumerate(curriculum):
    print(f"\n--- Training Phase {i+1}: {map_name} ({steps:,} steps) ---")

    if i > 0:
        # Create a new parallel environment for the next map
        new_env_kwargs = {"render_mode": None, "predefined_map": map_layout}
        new_vec_env = make_vec_env("standard", n_envs=CPUS, env_kwargs=new_env_kwargs)
        model.set_env(new_vec_env)

    # Pass the callback here!
    model.learn(
        total_timesteps=steps, 
        reset_num_timesteps=False, 
        callback=checkpoint_callback
    )

# 5. Save the final model
model.save("group9_ppo_agent_final")
print("\nTraining Complete! Saved as group9_ppo_agent_final.zip")
input("Press Enter to run visual test...")

# 6. Visual test (We use a single standard gym.make here so we can easily render it)
print("\nStarting Visual Test on Chokepoint map...")
env_human = gym.make("standard", render_mode="human", predefined_map=chokepoint_map)
obs, _ = env_human.reset()

for i in range(500):
    action, _ = model.predict(obs, deterministic=False) 
    print(f"Step {i}: Agent action {action}")
    obs, reward, done, trunc, info = env_human.step(action)
    time.sleep(0.1)

    if done or trunc:
        status = "Game Over" if info.get("game_over") else "Cleared or Timeout"
        coverage = info.get("total_covered_cells", "?")
        coverable = info.get("coverable_cells", "?")
        print(f"Episode ended! Status: {status} | Coverage: {coverage}/{coverable}")
        break

env_human.close()