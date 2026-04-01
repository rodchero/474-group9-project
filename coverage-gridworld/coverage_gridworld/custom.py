import numpy as np
import gymnasium as gym

OBS_MODE = "compass"      # Options: "compass", "local_3x3"
REWARD_MODE = "balanced"  # Options: "balanced", "cautious", "speed"

def observation_space(env: gym.Env) -> gym.spaces.Space:
    if OBS_MODE == "compass":
        # 5x5 local grid (25 cells, 7 colors)
        # + Target relative Y, X         (3 options each: -1/0/+1 mapped to 0/1/2)
        # + Enemy relative Y, X          (4 options: 0/1/2 = direction, 3 = no enemy)
        # + Danger flag                  (2 options: 0 or 1)
        # + Blocked directions L/D/R/U   (2 options each: 0 = free, 1 = blocked)
        return gym.spaces.MultiDiscrete([7] * 25 + [3, 3, 4, 4, 2, 2, 2, 2, 2])
    else:
        return gym.spaces.MultiDiscrete([7] * 9)


def observation(grid: np.ndarray):
    # 1. Locate the agent
    agent_mask = np.all(grid == [160, 161, 161], axis=-1)
    coords = np.argwhere(agent_mask)
    ay, ax = coords[0] if len(coords) > 0 else (0, 0)

    if OBS_MODE == "compass":
        # 2. Compress RGB grid to integer color IDs
        c_map = {
            (0,   0,   0):   0,  # Black       - unexplored
            (255, 255, 255): 1,  # White       - explored
            (101, 67,  33):  2,  # Brown       - wall
            (160, 161, 161): 3,  # Grey        - agent
            (31,  198, 0):   4,  # Green       - enemy
            (255, 0,   0):   5,  # Red         - enemy FOV (unexplored)
            (255, 127, 127): 6,  # Light red   - enemy FOV (explored)
        }
        comp = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                comp[i, j] = c_map.get(tuple(grid[i, j]), 0)

        # 3. Extract 5x5 local window padded with walls
        pad = np.pad(comp, 2, mode='constant', constant_values=2)
        cy, cx = ay + 2, ax + 2
        local_5x5 = pad[cy-2:cy+3, cx-2:cx+3].flatten()

        # 4. Target compass: direction to nearest unexplored (black) cell
        black_mask = np.all(grid == [0, 0, 0], axis=-1)
        blacks = np.argwhere(black_mask)
        if len(blacks) > 0:
            dist = np.sum(np.abs(blacks - [ay, ax]), axis=1)
            ty, tx = blacks[np.argmin(dist)]
            rel_y = int(np.sign(ty - ay) + 1)
            rel_x = int(np.sign(tx - ax) + 1)
        else:
            rel_y, rel_x = 1, 1

        # 5. Enemy compass: direction to nearest enemy (3 = no enemies)
        enemy_mask = np.all(grid == [31, 198, 0], axis=-1)
        enemies = np.argwhere(enemy_mask)
        if len(enemies) > 0:
            edist = np.sum(np.abs(enemies - [ay, ax]), axis=1)
            ey, ex = enemies[np.argmin(edist)]
            enemy_rel_y = int(np.sign(ey - ay) + 1)
            enemy_rel_x = int(np.sign(ex - ax) + 1)
        else:
            enemy_rel_y, enemy_rel_x = 3, 3

        # 6. Danger flag: any adjacent cell under enemy FOV?
        red_mask = np.all(grid == [255, 0, 0], axis=-1).astype(np.int8)
        pad_r = np.pad(red_mask, 1)
        danger = int(pad_r[ay:ay+3, ax:ax+3].sum() > 0)

        # 7. Blocked directions L/D/R/U
        # This is the critical signal for generalizing to unseen maps —
        # the agent explicitly knows which moves are valid right now
        # without having to memorize any layout.
        blocked = np.zeros(4, dtype=np.int64)
        for idx, (dy, dx) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            ny, nx = ay + dy, ax + dx
            if not (0 <= ny < 10 and 0 <= nx < 10):
                blocked[idx] = 1
            elif comp[ny, nx] in (2, 4):
                blocked[idx] = 1

        return np.concatenate(
            (local_5x5, [rel_y, rel_x, enemy_rel_y, enemy_rel_x, danger], blocked)
        ).astype(np.int64)

    else:
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
    if REWARD_MODE == "balanced":
        r = -0.3  # Base step penalty on all actions including STAY

        if info["new_cell_covered"]:
            cells_done = info["coverable_cells"] - info["cells_remaining"]
            progress = cells_done / max(info["coverable_cells"], 1)
            r += 10.0 + (15.0 * progress)  # +10 early, up to +25 late

        else:
            # Penalize hitting a wall: agent position didn't change but a move
            # action was taken. We detect this by checking if agent is still
            # on an already-explored (white or light-red) cell after moving.
            # The env returns new_cell_covered=False for both wall hits AND
            # revisiting explored cells, so we apply a moderate penalty for both.
            # This is the fix for the "action 3 UP into top wall" loop —
            # without this, hitting a wall costs only the base -0.3 step penalty
            # which is not enough signal to learn "don't do that on new maps".
            r -= 1.5

        if info["game_over"]:
            # Scale death penalty by cells remaining — dying early is much
            # worse than dying with 2 cells left
            r -= 50.0 + (info["cells_remaining"] * 1.5)

        if info["cells_remaining"] == 0:
            r += 1000.0 + (info["steps_remaining"] * 1.0)

    elif REWARD_MODE == "cautious":
        r = -0.3
        if info["new_cell_covered"]:
            r += 8.0
        else:
            r -= 1.0  # Wall hit / revisit penalty
        if info["game_over"]:
            r -= 150.0

    else:  # speed
        r = -1.0
        if info["new_cell_covered"]:
            r += 5.0
        if info["cells_remaining"] == 0:
            r += 300.0

    return float(r)