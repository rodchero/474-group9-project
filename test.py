import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import DQN
import time

# 1. The Target Map
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

env = gym.make("standard", render_mode=None, predefined_map=chokepoint_map)

# 2. Initialize a fresh DQN
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=128,
    gamma=0.99,
    
    # --- Dynamic Epsilon Control ---
    exploration_fraction=0.8,    # Decay ends at 800k (if total_timesteps=1M)
    exploration_initial_eps=1.0, # Start fully random
    exploration_final_eps=0.05,  # End at 5% random
    
    target_update_interval=5000,
)

# 3. Train purely on Chokepoint
print("Training Dedicated Chokepoint Agent...")
model.learn(total_timesteps=300000)

model.save("chokepoint_specialist")
print("\nSaved as chokepoint_specialist.zip")
input("Press Enter to watch it run...")

# 4. Visual Check
env_human = gym.make("standard", render_mode="human", predefined_map=chokepoint_map)
obs, _ = env_human.reset()

for i in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env_human.step(action)
    time.sleep(0.08) # Slowed down so you can watch the dodges!
    if done or trunc: 
        print(f"Result: {'Game Over' if info.get('game_over') else 'Cleared!'}")
        break

env_human.close()