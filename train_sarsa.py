import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import PPO

# 1. Define the Curriculum Maps (Copied from main.py)
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

# 2. Setup Environment with the Curriculum
# By passing a list, the environment rotates through them automatically
curriculum_maps = [maze_map, chokepoint_map]
env = gym.make("standard", render_mode=None, predefined_map_list=curriculum_maps)

# 3. Algorithm Implementation 
# Increased entropy (ent_coef) to force the agent to try running through the chokepoint
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.05, n_steps=2048, batch_size=64)

# 4. Training (Bumped up to give it time to learn the enemy rotations)
print("Training for the tournament...")
model.learn(total_timesteps=500000)

# 5. Save
model.save("group9_trained_agent")
print("Saved as group9_trained_agent.zip")
input("Run Visual")
# 6. Visual Validation on the Chokepoint Map specifically
env_human = gym.make("chokepoint", render_mode="human")
obs, _ = env_human.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env_human.step(action)
    if done or trunc: break
env_human.close()