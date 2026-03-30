import numpy as np
import gymnasium as gym

# --- EXPERIMENT TOGGLES ---
OBS_MODE = "feature_compressed"  # Options: "feature_compressed", "local_3x3"
REWARD_MODE = "cautious"         # Options: "standard", "cautious", "speed_run"

def observation_space(env: gym.Env) -> gym.spaces.Space:
    if OBS_MODE == "feature_compressed":
        # State array: [Up_cell, Down_cell, Left_cell, Right_cell, Nearest_Unexplored_Direction]
        # Cell values: 0 (Explored), 1 (Unexplored), 2 (Wall/Out-of-Bounds), 3 (Danger/Enemy)
        # Direction values: 0 (Left), 1 (Down), 2 (Right), 3 (Up), 4 (None)
        return gym.spaces.MultiDiscrete([4, 4, 4, 4, 5])
        
    elif OBS_MODE == "local_3x3":
        # A tiny 3x3 window around the agent (9 cells, 7 possible colors)
        return gym.spaces.MultiDiscrete([7] * 9)

def observation(grid: np.ndarray):
    # Find the agent's coordinates
    agent_mask = np.all(grid == [160, 161, 161], axis=-1)
    coords = np.argwhere(agent_mask)
    if len(coords) == 0:
        agent_y, agent_x = 0, 0
    else:
        agent_y, agent_x = coords[0]

    if OBS_MODE == "feature_compressed":
        def get_cell_type(y, x):
            if y < 0 or y >= 10 or x < 0 or x >= 10: return 2 # Treat bounds as Walls
            color = tuple(grid[y, x])
            if color == (0, 0, 0): return 1                    # Unexplored
            if color in [(255, 255, 255), (160, 161, 161)]: return 0 # Explored / Agent
            if color == (101, 67, 33): return 2                # Wall
            return 3                                           # Danger (Enemy / FOV)

        up_val = get_cell_type(agent_y - 1, agent_x)
        down_val = get_cell_type(agent_y + 1, agent_x)
        left_val = get_cell_type(agent_y, agent_x - 1)
        right_val = get_cell_type(agent_y, agent_x + 1)

        # Find the general direction of the nearest black cell
        black_mask = np.all(grid == [0, 0, 0], axis=-1)
        black_coords = np.argwhere(black_mask)
        
        nearest_dir = 4
        if len(black_coords) > 0:
            # Manhattan distance
            distances = np.abs(black_coords[:, 0] - agent_y) + np.abs(black_coords[:, 1] - agent_x)
            target_y, target_x = black_coords[np.argmin(distances)]
            
            if abs(target_y - agent_y) > abs(target_x - agent_x):
                nearest_dir = 3 if target_y < agent_y else 1 # Up or Down
            else:
                nearest_dir = 0 if target_x < agent_x else 2 # Left or Right
                
        return np.array([up_val, down_val, left_val, right_val, nearest_dir], dtype=np.int64)

    elif OBS_MODE == "local_3x3":
        # Compresses RGB to integer IDs for a tight 3x3 grid
        color_map = {(0,0,0):0, (255,255,255):1, (101,67,33):2, (160,161,161):3, (31,198,0):4, (255,0,0):5, (255,127,127):6}
        compressed = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                compressed[i, j] = color_map.get(tuple(grid[i, j]), 0)
                
        padded = np.pad(compressed, pad_width=1, mode='constant', constant_values=2)
        cy, cx = agent_y + 1, agent_x + 1
        return padded[cy-1:cy+2, cx-1:cx+2].flatten()

def reward(info: dict) -> float:
    r = 0.0
    if REWARD_MODE == "standard":
        r -= 0.5                            # Time penalty
        if info["new_cell_covered"]: r += 15.0
        if info["game_over"]: r -= 100.0
        if info["cells_remaining"] == 0: r += 100.0
            
    elif REWARD_MODE == "cautious":
        r -= 0.1
        if info["new_cell_covered"]: r += 10.0
        if info["game_over"]: r -= 200.0    # Massive penalty for dying
        
    elif REWARD_MODE == "speed_run":
        r -= 2.0                            # Heavy time penalty to force rushing
        if info["new_cell_covered"]: r += 5.0
        if info["game_over"]: r -= 50.0
        if info["cells_remaining"] == 0: r += 200.0
        
    return r