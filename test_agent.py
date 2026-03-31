import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import DQN
import time

# 1. Load the trained DQN model
# Make sure this matches the name you saved in your train_dqn.py script
print("Loading saved model...")
model = DQN.load("group9_dqn_agent")
print("Model loaded successfully!")

# 2. Setup the environment for testing
# You can change "chokepoint" to "sneaky_enemies", "maze", or "safe"
map_name = "maze"
env = gym.make(map_name, render_mode="human", activate_game_status=True)

# 3. Run the evaluation loop
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    # deterministic=True forces the agent to take the "best" action it learned,
    # which means it will pick the action with the absolute highest Q-value.
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    
    # Optional: Slow down the rendering slightly so you can watch its decision-making
    time.sleep(0.05)

# 4. Print final stats
print("\n--- Episode Finished ---")
print(f"Map: {map_name}")
print(f"Total Reward: {total_reward:.2f}")
print(f"Coverage: {info['total_covered_cells']} / {info['coverable_cells']} cells")
if info['game_over']:
    print("Status: SPOTTED BY ENEMY")
elif info['cells_remaining'] == 0:
    print("Status: MAP CLEARED!")

env.close()