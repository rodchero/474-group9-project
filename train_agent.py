import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import DQN
import time

# 1. Curriculum Maps
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

# Set up the curriculum list
curriculum = [
    ("Maze", maze_map),
    ("Chokepoint", chokepoint_map)
]

# Initialize the first environment
first_env = gym.make("standard", render_mode=None, predefined_map=curriculum[0][1])

# 2. Initialize DQN
model = DQN(
    "MlpPolicy",
    first_env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,          
    learning_starts=1000,        
    batch_size=64,
    gamma=0.99,                 
    exploration_fraction=0.5,    
    exploration_initial_eps=1.0, 
    exploration_final_eps=0.05,  
    target_update_interval=1000  
)

# 3. Train across the curriculum
steps_per_map = 25000  # 250k on Maze, then 250k on Chokepoint (500k total)

for i, (map_name, map_layout) in enumerate(curriculum):
    print(f"\n--- Training Phase {i+1}: {map_name} Map ---")
    
    if i > 0:
        # Swap the environment for the new map
        new_env = gym.make("standard", render_mode=None, predefined_map=map_layout)
        model.set_env(new_env)
        
    # reset_num_timesteps=False is CRITICAL here so the exploration decay doesn't reset
    model.learn(total_timesteps=steps_per_map, reset_num_timesteps=False)

# 4. Save
model.save("group9_dqn_agent")
print("\nSaved as group9_dqn_agent.zip")
input("Press Enter to Run Visuals...")

# 5. Visual Check (Testing on Chokepoint to see if it learned the hard part)
print("\nStarting Visual Test...")
env_human = gym.make("standard", render_mode="human", predefined_map=chokepoint_map)
obs, _ = env_human.reset()

for i in range(500):
    # Predict the best action
    action, _ = model.predict(obs, deterministic=True)
    
    # Print what the agent is actually trying to do
    print(f"Step {i}: Agent attempting action {action}")
    
    obs, reward, done, trunc, info = env_human.step(action)
    
    # Force the game to pause for a fraction of a second so you can watch
    time.sleep(0.1) 
    
    if done or trunc: 
        print(f"Episode ended! Status: {'Game Over' if info.get('game_over') else 'Cleared or Timeout'}")
        break

env_human.close()