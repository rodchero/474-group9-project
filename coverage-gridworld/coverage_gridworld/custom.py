import numpy as np
import gymnasium as gym

OBS_MODE = "compass"      # Options: "compass", "local_3x3"
REWARD_MODE = "balanced"  # Options: "balanced", "cautious", "speed"

def observation_space(env: gym.Env) -> gym.spaces.Space:
    if OBS_MODE == "compass":
        return gym.spaces.MultiDiscrete([7] * 25 + [3, 3, 3, 3, 2])
    else:
        return gym.spaces.MultiDiscrete([7] * 9)

def observation(grid: np.ndarray):
    agent_mask = np.all(grid == [160, 161, 161], axis=-1)
    coords = np.argwhere(agent_mask)
    ay, ax = coords[0] if len(coords) > 0 else (0, 0)

    if OBS_MODE == "compass":
        c_map = {(0,0,0):0, (255,255,255):1, (101,67,33):2, (160,161,161):3,
                 (31,198,0):4, (255,0,0):5, (255,127,127):6}
        comp = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                comp[i,j] = c_map.get(tuple(grid[i,j]), 0)

        pad = np.pad(comp, 2, mode='constant', constant_values=2)
        cy, cx = ay + 2, ax + 2
        local_5x5 = pad[cy-2:cy+3, cx-2:cx+3].flatten()

        black_mask = np.all(grid == [0, 0, 0], axis=-1)
        blacks = np.argwhere(black_mask)
        if len(blacks) > 0:
            dist = np.sum(np.abs(blacks - [ay, ax]), axis=1)
            ty, tx = blacks[np.argmin(dist)]
            rel_y = int(np.sign(ty - ay) + 1)
            rel_x = int(np.sign(tx - ax) + 1)
        else:
            rel_y, rel_x = 1, 1

        enemy_mask = np.all(grid == [31, 198, 0], axis=-1)
        enemies = np.argwhere(enemy_mask)
        if len(enemies) > 0:
            edist = np.sum(np.abs(enemies - [ay, ax]), axis=1)
            ey, ex = enemies[np.argmin(edist)]
            enemy_rel_y = int(np.sign(ey - ay) + 1)
            enemy_rel_x = int(np.sign(ex - ax) + 1)
        else:
            enemy_rel_y, enemy_rel_x = 1, 1

        red_mask = np.all(grid == [255, 0, 0], axis=-1).astype(np.int8)
        pad_r = np.pad(red_mask, 1)
        danger = int(pad_r[ay:ay+3, ax:ax+3].sum() > 0)

        return np.concatenate(
            (local_5x5, [rel_y, rel_x, enemy_rel_y, enemy_rel_x, danger])
        ).astype(np.int64)

    else:
        c_map = {(0,0,0):0, (255,255,255):1, (101,67,33):2, (160,161,161):3,
                 (31,198,0):4, (255,0,0):5, (255,127,127):6}
        comp = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                comp[i,j] = c_map.get(tuple(grid[i,j]), 0)

        pad = np.pad(comp, 1, mode='constant', constant_values=2)
        cy, cx = ay + 1, ax + 1
        return pad[cy-1:cy+2, cx-1:cx+2].flatten().astype(np.int64)


def reward(info: dict) -> float:
    if REWARD_MODE == "balanced":
        # Base step penalty applies to ALL actions including STAY,
        # so freezing is never a free escape
        r = -0.5

        if info["new_cell_covered"]:
            # Reward scales with progress so late cells stay valuable
            cells_done = info["coverable_cells"] - info["cells_remaining"]
            progress = cells_done / max(info["coverable_cells"], 1)
            r += 10.0 + (15.0 * progress)  # +10 early, up to +25 late
        
        # NOTE: removed the -2.0 revisit penalty entirely — it was causing
        # STAY to become the dominant strategy since STAY never triggers it.
        # The step penalty alone is sufficient to discourage loitering.

        if info["game_over"]:
            r -= 100.0  # Flat death penalty, simple and stable

        if info["cells_remaining"] == 0:
            r += 500.0 + (info["steps_remaining"] * 1.0)

    elif REWARD_MODE == "cautious":
        r = -0.5
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