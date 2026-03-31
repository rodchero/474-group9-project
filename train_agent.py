import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import DQN
import time

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

# order of maps to train on
curriculum = [
    ("JustGo",     just_go_map,     150_000),
    ("Safe",       safe_map,        200_000),
    ("Maze",       maze_map,        300_000),
    ("Chokepoint", chokepoint_map,  500_000),
]

total_steps = sum(s for _, _, s in curriculum)  # 650k total

# 3. Initialize the first environment
first_env = gym.make("standard", render_mode=None, predefined_map=curriculum[0][1])

# 4. Initialize DQN
model = DQN(
    "MlpPolicy",
    first_env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,

    # Start learning sooner so early curriculum episodes aren't wasted
    learning_starts=1_000,
    batch_size=128,
    gamma=0.99,

    # Exploration: decays over 80% of total training — ensures late-stage maps still explore
    exploration_fraction=0.8,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,

    # Stability: more frequent target updates for better Q-value estimates early on
    target_update_interval=1_000,
    train_freq=4,
    gradient_steps=1
)

# 5. Train across the curriculum
for i, (map_name, map_layout, steps) in enumerate(curriculum):
    print(f"\n--- Training Phase {i+1}: {map_name} ({steps:,} steps) ---")

    if i > 0:
        new_env = gym.make("standard", render_mode=None, predefined_map=map_layout)
        model.set_env(new_env)

    # reset_num_timesteps=False is critical — keeps exploration decay continuous
    # across all curriculum phases rather than resetting epsilon each time
    model.learn(total_timesteps=steps, reset_num_timesteps=False)

# 6. Save
model.save("group9_dqn_agent")
print("\nSaved as group9_dqn_agent.zip")
input("Press Enter to run visual test...")

# 7. Visual test on the hardest map
print("\nStarting Visual Test on Chokepoint map...")
env_human = gym.make("standard", render_mode="human", predefined_map=chokepoint_map)
obs, _ = env_human.reset()

for i in range(500):
    action, _ = model.predict(obs, deterministic=True)
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