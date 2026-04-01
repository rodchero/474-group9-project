import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import DQN
import time

# Load the FINAL model, not the EvalCallback best_model checkpoint.
# best_model is saved based on chokepoint performance specifically — it tends
# to be over-specialized. The final model has seen all curriculum phases and
# generalizes better to unseen maps.
print("Loading final model...")
model = DQN.load("group9_dqn_agent")
print("Model loaded successfully!")

# Change this to test different maps: "safe", "maze", "chokepoint", "just_go"
map_name = "safe"
env = gym.make(map_name, render_mode="human", activate_game_status=True)

obs, _ = env.reset()

# Sanity check: confirm blocked signal is correct at spawn position
# Agent always starts at (0,0) — UP and LEFT should always be blocked
print(f"\n--- Observation sanity check at spawn ---")
print(f"  blocked_L={obs[30]}, blocked_D={obs[31]}, blocked_R={obs[32]}, blocked_U={obs[33]}")
print(f"  Expected: blocked_L=1 (edge), blocked_U=1 (edge)")
print(f"  target_compass: rel_y={obs[25]}, rel_x={obs[26]}")
print(f"  enemy_compass:  rel_y={obs[27]}, rel_x={obs[28]}  (3=no enemy)")
print(f"  danger_flag={obs[29]}")
print()

done = False
total_reward = 0
step_count = 0
action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP", 4: "STAY"}

while not done:
    action, _states = model.predict(obs, deterministic=True)

    # Debug: print what the agent is doing and why each step
    print(f"Step {step_count:3d}: action={action_names[int(action)]:5s} | "
          f"target=({obs[25]},{obs[26]}) enemy=({obs[27]},{obs[28]}) "
          f"danger={obs[29]} blocked_LDRU=({obs[30]}{obs[31]}{obs[32]}{obs[33]})")

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1

    time.sleep(0.08)

print("\n--- Episode Finished ---")
print(f"Total Reward:  {total_reward:.2f}")
print(f"Coverage:      {info['total_covered_cells']} / {info['coverable_cells']} cells")
print(f"Steps taken:   {step_count}")
if info['game_over']:
    print("Status: SPOTTED BY ENEMY")
elif info['cells_remaining'] == 0:
    print("Status: MAP CLEARED!")
else:
    print("Status: TIMEOUT")

env.close()