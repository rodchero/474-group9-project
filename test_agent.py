import numpy as np
import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import PPO
import time

# --- Observation index reference (Matches custom.py exactly) ---
# [0:100]   full 10x10 grid flattened (100 cells, 7 color IDs)
# [100]     target rel_y
# [101]     target rel_x
# [102]     enemy rel_y  (3 = no enemy)
# [103]     enemy rel_x  (3 = no enemy)
# [104]     danger flag
# [105]     fov pressure (0-4 adjacent red cells)
# [106]     blocked LEFT
# [107]     blocked DOWN
# [108]     blocked RIGHT
# [109]     blocked UP

IDX_TARGET_Y  = 100
IDX_TARGET_X  = 101
IDX_ENEMY_Y   = 102
IDX_ENEMY_X   = 103
IDX_DANGER    = 104
IDX_FOV_PRESS = 105
IDX_BLOCK_L   = 106
IDX_BLOCK_D   = 107
IDX_BLOCK_R   = 108
IDX_BLOCK_U   = 109

print("Loading final model...")
model = PPO.load("./ppo_checkpoints\ppo_agent_1200000_steps.zip")
# model = PPO.load("group9_ppo_agent")
print("Model loaded successfully!")

# Change to test different maps: "safe", "maze", "chokepoint", "just_go"
map_name = "chokepoint"

# Create the pure environment (No wrappers needed!)
env = gym.make(map_name, render_mode="human", activate_game_status=True)

obs, _ = env.reset()

# --- Sanity check at spawn ---
print(f"\n--- Observation sanity check at spawn ---")
print(f"  blocked_L={obs[IDX_BLOCK_L]}, blocked_D={obs[IDX_BLOCK_D]}, "
      f"blocked_R={obs[IDX_BLOCK_R]}, blocked_U={obs[IDX_BLOCK_U]}")
print(f"  Expected: blocked_L=1 (grid edge), blocked_U=1 (grid edge)")
print(f"  target_compass: rel_y={obs[IDX_TARGET_Y]}, rel_x={obs[IDX_TARGET_X]}")
print(f"  enemy_compass:  rel_y={obs[IDX_ENEMY_Y]}, rel_x={obs[IDX_ENEMY_X]}  (3=no enemy)")
print(f"  danger_flag={obs[IDX_DANGER]}")
print(f"  fov_pressure={obs[IDX_FOV_PRESS]}")
print()

done = False
total_reward = 0
step_count = 0
action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP", 4: "STAY"}

while not done:
    # PPO trained on VecEnv expects a batched observation
    action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
    action = int(action[0])

    print(f"Step {step_count:3d}: action={action_names[action]:5s} | "
          f"target=({obs[IDX_TARGET_Y]},{obs[IDX_TARGET_X]}) "
          f"enemy=({obs[IDX_ENEMY_Y]},{obs[IDX_ENEMY_X]}) "
          f"danger={obs[IDX_DANGER]} "
          f"fov_press={obs[IDX_FOV_PRESS]} "
          f"blocked=({obs[IDX_BLOCK_L]}{obs[IDX_BLOCK_D]}{obs[IDX_BLOCK_R]}{obs[IDX_BLOCK_U]})")

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1

    time.sleep(0.08)

print("\n--- Episode Finished ---")
print(f"Total Reward:  {total_reward:.2f}")
print(f"Coverage:      {info['total_covered_cells']} / {info['coverable_cells']} cells")
print(f"Steps taken:   {step_count}")
if info.get('game_over'):
    print("Status: SPOTTED BY ENEMY")
elif info.get('cells_remaining') == 0:
    print("Status: MAP CLEARED!")
else:
    print("Status: TIMEOUT")

env.close()