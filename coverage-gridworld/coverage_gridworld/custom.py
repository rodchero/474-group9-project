import numpy as np
import gymnasium as gym

OBS_MODE = "advanced"      

def observation_space(env: gym.Env) -> gym.spaces.Space:
    if OBS_MODE == "compass":
        # 5x5 local grid (25 cells, 7 colors)
        # + Target relative Y, X         (3 options each: -1/0/+1 mapped to 0/1/2)
        # + Enemy relative Y, X          (4 options: 0/1/2 = direction, 3 = no enemy)
        # + Danger flag                  (2 options: 0 or 1)
        # + Blocked directions L/D/R/U   (2 options each: 0 = free, 1 = blocked)
        return gym.spaces.MultiDiscrete([7] * 25 + [3, 3, 4, 4, 2, 2, 2, 2, 2])
    elif OBS_MODE == "advanced":
        # 5x5 local grid (25) + exact rel_y, rel_x (21, 21) + exact enemy_y, enemy_x (21, 21) + 4 danger directions
        return gym.spaces.MultiDiscrete([7] * 25 + [21, 21, 21, 21, 2, 2, 2, 2])
    else:
        return gym.spaces.MultiDiscrete([7] * 9)


def observation(grid: np.ndarray):
    # 1. Locate the agent
    agent_mask = np.all(grid == [160, 161, 161], axis=-1)
    coords = np.argwhere(agent_mask)
    ay, ax = coords[0] if len(coords) > 0 else (0, 0)

    if OBS_MODE == "advanced":
        c_map = {(0,0,0):0, (255,255,255):1, (101,67,33):2, (160,161,161):3,
                 (31,198,0):4, (255,0,0):5, (255,127,127):6}
        comp = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                comp[i,j] = c_map.get(tuple(grid[i,j]), 0)

        # 1. 5x5 Local Grid
        pad = np.pad(comp, 2, mode='constant', constant_values=2)
        cy, cx = ay + 2, ax + 2
        local_5x5 = pad[cy-2:cy+3, cx-2:cx+3].flatten()

        # 2. Exact Delta to nearest Black cell (Mapped from [-10, 10] to [0, 20])
        black_mask = np.all(grid == [0, 0, 0], axis=-1)
        blacks = np.argwhere(black_mask)
        if len(blacks) > 0:
            dist = np.sum(np.abs(blacks - [ay, ax]), axis=1)
            ty, tx = blacks[np.argmin(dist)]
            rel_y = max(-10, min(10, ty - ay)) + 10
            rel_x = max(-10, min(10, tx - ax)) + 10
        else:
            rel_y, rel_x = 10, 10 # Center/Zero delta

        # 3. Exact Delta to nearest Enemy
        enemy_mask = np.all(grid == [31, 198, 0], axis=-1)
        enemies = np.argwhere(enemy_mask)
        if len(enemies) > 0:
            edist = np.sum(np.abs(enemies - [ay, ax]), axis=1)
            ey, ex = enemies[np.argmin(edist)]
            enemy_rel_y = max(-10, min(10, ey - ay)) + 10
            enemy_rel_x = max(-10, min(10, ex - ax)) + 10
        else:
            enemy_rel_y, enemy_rel_x = 10, 10

        # 4. Immediate Directional Danger (Up, Down, Left, Right)
        # Danger = Red (5), Light Red (6), or Enemy Green (4)
        def is_danger(y, x):
            if 0 <= y < 10 and 0 <= x < 10:
                return 1 if comp[y, x] in [4, 5, 6] else 0
            return 0 # Out of bounds (walls are safe)

        danger_up = is_danger(ay - 1, ax)
        danger_down = is_danger(ay + 1, ax)
        danger_left = is_danger(ay, ax - 1)
        danger_right = is_danger(ay, ax + 1)

        return np.concatenate(
            (local_5x5, [rel_y, rel_x, enemy_rel_y, enemy_rel_x, danger_up, danger_down, danger_left, danger_right])
        ).astype(np.int64)
    
    elif OBS_MODE == "compass":
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
    # Flat step penalty applied EVERY turn, even if the agent uses STAY
    r = -0.1 

    if info["new_cell_covered"]:
        # Dynamic reward based on progress
        progress = (info["coverable_cells"] - info["cells_remaining"]) / max(info["coverable_cells"], 1)
        r += 10.0 + (15.0 * progress) 

    if info["game_over"]:
        # Make sure dying is incredibly punishing so it doesn't just suicide to end the episode
        # but not so punishing that it refuses to take risks? idk man 
        r -= 100.0 

    if info["cells_remaining"] == 0:
        r += 500.0 + (info["steps_remaining"] * 0.5)

    return float(r)