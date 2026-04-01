import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import DQN

# 1. Define the Advanced Curriculum Maps
# We skip the easy maps here so the agent focuses purely on hard timing and complex walls.
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

sneaky_enemies_map = [
    [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
]

safe = [
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

advanced_curriculum = [safe]
map_name = "safe"
# 2. Setup the New Training Environment
env = gym.make(map_name, render_mode=None, predefined_map_list=advanced_curriculum)

# 3. Load the Existing Model and Attach the New Environment
print("Loading previously trained agent...")
# Make sure the filename matches your saved model (no .zip needed)
model = DQN.load("group9_dqn_agent", env=env)
print("Agent loaded successfully! Starting fine-tuning...")

# 4. Resume Training
# Because it already knows the basics (how to use the compass, avoid walls), 
# it will spend these timesteps purely mastering the complex enemy layouts.
model.learn(total_timesteps=100000)

# 5. Save the Upgraded Model
# We save it under a new name so you don't overwrite your original backup
model.save("group9_trained_agent_v2")
print("Upgraded agent saved as group9_trained_agent_v2.zip")
input("Run Visual")
# 6. Visual Validation on the final boss map
env_human = gym.make(map_name, render_mode="human")
obs, _ = env_human.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env_human.step(action)
    if done or trunc: break
env_human.close()