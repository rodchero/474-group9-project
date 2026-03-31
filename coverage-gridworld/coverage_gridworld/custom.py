import numpy as np
import gymnasium as gym

# --- TO SWITCH MODES FOR EXPERIMENTS ---
OBS_MODE = "compass"      # Options: "compass", "local_3x3"
REWARD_MODE = "speed"  # Options: "balanced", "cautious", "speed"

def observation_space(env: gym.Env) -> gym.spaces.Space:
    if OBS_MODE == "compass":
        # 5x5 local grid (25 cells) + Target_Rel_Y (1) + Target_Rel_X (1)
        # We use 7 for the color IDs, and 3 for the relative directions [-1, 0, 1] mapped to [0, 1, 2]
        return gym.spaces.MultiDiscrete([7] * 25 + [3, 3])
    else:
        # Standard 3x3 local grid (9 cells, 7 possible color IDs)
        return gym.spaces.MultiDiscrete([7] * 9)

def observation(grid: np.ndarray):
    # 1. Locate the agent's current coordinates
    agent_mask = np.all(grid == [160, 161, 161], axis=-1)
    coords = np.argwhere(agent_mask)
    ay, ax = coords[0] if len(coords) > 0 else (0, 0)

    if OBS_MODE == "compass":
        # 2. Compress the RGB grid into integer IDs
        c_map = {(0,0,0):0, (255,255,255):1, (101,67,33):2, (160,161,161):3, (31,198,0):4, (255,0,0):5, (255,127,127):6}
        comp = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10): 
                comp[i,j] = c_map.get(tuple(grid[i,j]), 0)
        
        # 3. Extract the 5x5 local window
        # Pad with '2' (Walls) so we can slice a 5x5 grid even when the agent is against the edge of the map
        pad = np.pad(comp, 2, mode='constant', constant_values=2)
        cy, cx = ay + 2, ax + 2
        local_5x5 = pad[cy-2:cy+3, cx-2:cx+3].flatten()

        # 4. Compass logic: find the Manhattan distance to all unexplored (black) cells
        black_mask = np.all(grid == [0, 0, 0], axis=-1)
        blacks = np.argwhere(black_mask)
        
        if len(blacks) > 0:
            dist = np.sum(np.abs(blacks - [ay, ax]), axis=1)
            ty, tx = blacks[np.argmin(dist)]
            # Map the relative direction from [-1, 0, 1] to [0, 1, 2] so it fits in MultiDiscrete
            rel_y, rel_x = np.sign(ty - ay) + 1, np.sign(tx - ax) + 1
        else:
            rel_y, rel_x = 1, 1 # No goals left, default to neutral

        # 5. Combine the local vision array with the compass array
        return np.concatenate((local_5x5, [rel_y, rel_x])).astype(np.int64)

    else:
        # --- LOCAL 3x3 LOGIC (For your required second observation space) ---
        c_map = {(0,0,0):0, (255,255,255):1, (101,67,33):2, (160,161,161):3, (31,198,0):4, (255,0,0):5, (255,127,127):6}
        comp = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10): 
                comp[i,j] = c_map.get(tuple(grid[i,j]), 0)
        
        # Pad by 1 for a 3x3 window
        pad = np.pad(comp, 1, mode='constant', constant_values=2)
        cy, cx = ay + 1, ax + 1
        return pad[cy-1:cy+2, cx-1:cx+2].flatten()

def reward(info: dict) -> float:
    if REWARD_MODE == "balanced":
        r = -0.1 # Small step penalty
        if info["new_cell_covered"]: r += 10.0
        if info["game_over"]: r -= 50.0
        if info["cells_remaining"] == 0: r += 100.0
    elif REWARD_MODE == "cautious":
        r = -0.05
        if info["new_cell_covered"]: r += 5.0
        if info["game_over"]: r -= 200.0 # Heavy death penalty
    else: # speed
        r = -1.0 # Heavy step penalty forces speed
        if info["new_cell_covered"]: r += 2.0
        if info["cells_remaining"] == 0: r += 500.0
    return float(r)