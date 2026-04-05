import numpy as np
import gymnasium as gym

OBS_MODE = "compass"      # Options: "compass", "local_3x3"
REWARD_MODE = "balanced"  # Options: "balanced", "cautious", "speed"

CURRENT_DANGER_LEVEL = 0

def observation_space(env: gym.Env) -> gym.spaces.Space:
    if OBS_MODE == "compass":
        # Full 10x10 grid (100 cells, 7 colors)
        # + Target relative Y, X              (3 options each)
        # + Enemy relative Y, X               (4 options: 0/1/2 = direction, 3 = no enemy)
        # + Danger flag                        (2 options: 0/1)
        # + FOV pressure: red cells adjacent  (5 options: 0-4, how many of 4 dirs are red)
        # + Blocked directions L/D/R/U        (2 options each)
        # Total: 100 + 2 + 2 + 2 + 1 + 4 = 111 values
        return gym.spaces.MultiDiscrete([7] * 100 + [3, 3, 4, 4, 2, 5, 2, 2, 2, 2])
    else:
        return gym.spaces.MultiDiscrete([7] * 9)


def observation(grid: np.ndarray):
    # 1. Locate the agent
    agent_mask = np.all(grid == [160, 161, 161], axis=-1)
    coords = np.argwhere(agent_mask)
    ay, ax = coords[0] if len(coords) > 0 else (0, 0)

    if OBS_MODE == "compass":
        # 2. Compress full 10x10 RGB grid to integer color IDs
        c_map = {
            (0,   0,   0):   0,  # Black      - unexplored
            (255, 255, 255): 1,  # White      - explored
            (101, 67,  33):  2,  # Brown      - wall
            (160, 161, 161): 3,  # Grey       - agent
            (31,  198, 0):   4,  # Green      - enemy
            (255, 0,   0):   5,  # Red        - enemy FOV (unexplored)
            (255, 127, 127): 6,  # Light red  - enemy FOV (explored)
        }
        comp = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                comp[i, j] = c_map.get(tuple(grid[i, j]), 0)

        full_grid = comp.flatten()

        # 3. Target compass: direction to nearest unexplored (black) cell
        black_mask = np.all(grid == [0, 0, 0], axis=-1)
        blacks = np.argwhere(black_mask)
        if len(blacks) > 0:
            dist = np.sum(np.abs(blacks - [ay, ax]), axis=1)
            ty, tx = blacks[np.argmin(dist)]
            rel_y = int(np.sign(ty - ay) + 1)
            rel_x = int(np.sign(tx - ax) + 1)
        else:
            rel_y, rel_x = 1, 1

        # 4. Enemy compass: direction to nearest enemy (3 = no enemies)
        enemy_mask = np.all(grid == [31, 198, 0], axis=-1)
        enemies = np.argwhere(enemy_mask)
        if len(enemies) > 0:
            edist = np.sum(np.abs(enemies - [ay, ax]), axis=1)
            ey, ex = enemies[np.argmin(edist)]
            enemy_rel_y = int(np.sign(ey - ay) + 1)
            enemy_rel_x = int(np.sign(ex - ax) + 1)
        else:
            enemy_rel_y, enemy_rel_x = 3, 3

        # 5. Danger flag: is the agent's current cell under FOV?
        # True if agent is standing on a red/light-red cell
        agent_cell = comp[ay, ax]
        danger = int(agent_cell in (5, 6))

        # 6. FOV pressure: how many of the 4 adjacent cells are red/light-red?
        # This is the enemy timing signal — it tells the agent how "surrounded"
        # by guard vision it is right now. As a guard rotates away, this count
        # drops from 2 → 1 → 0, teaching the agent to wait for 0 before moving
        # into a previously-watched cell.
        fov_pressure = 0
        for dy, dx in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            ny, nx = ay + dy, ax + dx
            if 0 <= ny < 10 and 0 <= nx < 10:
                if comp[ny, nx] in (5, 6):
                    fov_pressure += 1

        global CURRENT_DANGER_LEVEL
        CURRENT_DANGER_LEVEL = fov_pressure
        # 7. Blocked directions L/D/R/U
        # Wall or enemy in that direction = 1, free = 0.
        # This is computed purely from the current grid state — no wrapper needed.
        blocked = np.zeros(4, dtype=np.int64)
        for idx, (dy, dx) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            ny, nx = ay + dy, ax + dx
            if not (0 <= ny < 10 and 0 <= nx < 10):
                blocked[idx] = 1  # Grid edge = wall
            elif comp[ny, nx] in (2, 4):
                blocked[idx] = 1  # Brown wall or green enemy = blocked

        return np.concatenate(
            (full_grid, [rel_y, rel_x, enemy_rel_y, enemy_rel_x,
                         danger, fov_pressure], blocked)
        ).astype(np.int64)

    else:
        # --- LOCAL 3x3 (second observation space for report experiments) ---
        c_map = {
            (0,   0,   0):   0,
            (255, 255, 255): 1,
            (101, 67,  33):  2,
            (160, 161, 161): 3,
            (31,  198, 0):   4,
            (255, 0,   0):   5,
            (255, 127, 127): 6,
        }
        comp = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                comp[i, j] = c_map.get(tuple(grid[i, j]), 0)

        pad = np.pad(comp, 1, mode='constant', constant_values=2)
        cy, cx = ay + 1, ax + 1
        return pad[cy-1:cy+2, cx-1:cx+2].flatten().astype(np.int64)


def reward(info: dict) -> float:

    global CURRENT_DANGER_LEVEL

    if REWARD_MODE == "balanced":
        if CURRENT_DANGER_LEVEL > 0:
            r = -0.1  # near danger slow down
        else:
            r = -0.6  # not near danger, keep moving

        if info["new_cell_covered"]:
            cells_done = info["coverable_cells"] - info["cells_remaining"]
            progress = cells_done / max(info["coverable_cells"], 1)
            
            # The base reward (scales from 10 to 25)
            base_reward = 10.0 + (15.0 * progress)  
            
            # # HAZARD PAY: +5 bonus points for every level of FOV pressure
            # hazard_bonus = CURRENT_DANGER_LEVEL * 5.0 
            
            r += base_reward

        if info["game_over"]:
            r -= 50.0 + (info["cells_remaining"] * 1.5)

        if info["cells_remaining"] == 0:
            r += 1000.0 + (info["steps_remaining"] * 1.0)

    elif REWARD_MODE == "cautious":
        r = -0.3
        if info["new_cell_covered"]:
            r += 8.0
        if info["game_over"]:
            r -= 150.0

    else:  # speed
        r = -1.0
        if info["new_cell_covered"]:
            r += 5.0
        if info["cells_remaining"] == 0:
            r += 300.0

    return float(r)