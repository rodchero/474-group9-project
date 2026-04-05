import numpy as np
import gymnasium as gym

REWARD_STRUCTURE = "R1" # R1, R2, else speed reward structure
OBSERVATION_STRUCTURE = "O1" #O1 else O2

CURRENT_DANGER_LEVEL = 0

def observation_space(env: gym.Env) -> gym.spaces.Space:
    if OBSERVATION_STRUCTURE == "O1":
        # 5x5 local grid (25) + exact rel_y, rel_x (21, 21) + exact enemy_y, enemy_x (21, 21) + 4 danger directions
        return gym.spaces.MultiDiscrete(
                [7] * 25 + 
                [21, 21] + 
                [21, 21] + 
                [2, 2, 2, 2] + 
                [2] * 100  
            )
    else:
        # Full 10x10 grid (100 cells, 7 colors)
        # + Target relative Y, X              (3 options each)
        # + Enemy relative Y, X               (4 options: 0/1/2 = direction, 3 = no enemy)
        # + Danger flag                        (2 options: 0/1)
        # + FOV pressure: red cells adjacent  (5 options: 0-4, how many of 4 dirs are red)
        # + Blocked directions L/D/R/U        (2 options each)
        # Total: 100 + 2 + 2 + 2 + 1 + 4 = 111 values
        return gym.spaces.MultiDiscrete([7] * 100 + [3, 3, 4, 4, 2, 5, 2, 2, 2, 2])


def observation(grid: np.ndarray):
    
    agent_mask = np.all(grid == [160, 161, 161], axis=-1)
    coords = np.argwhere(agent_mask)
    ay, ax = coords[0] if len(coords) > 0 else (0, 0)

    if OBSERVATION_STRUCTURE == "O1":

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

        # 5. NEW: Global Unvisited Map (100 features)
        # Convert the black_mask (True/False for unvisited cells) into 1s and 0s, and flatten it
        black_mask = np.all(grid == [0, 0, 0], axis=-1)
        global_unvisited = black_mask.flatten().astype(np.int64)



        return np.concatenate((
                local_5x5, 
                [rel_y, rel_x, enemy_rel_y, enemy_rel_x, danger_up, danger_down, danger_left, danger_right], 
                global_unvisited
            )).astype(np.int64)

    else:
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


def reward(info: dict) -> float:
    
    if REWARD_STRUCTURE == "R1": # first reward structure type
        r = -0.1  # Standard step penalty

        if info["new_cell_covered"]:
            progress = (info["coverable_cells"] - info["cells_remaining"]) / max(info["coverable_cells"], 1)
            
            # THE ENDGAME SPIKE
            if info["cells_remaining"] <= 10:
                # Desperation mode: The final cells are worth an absolute fortune. 
                # The agent will gladly risk a death penalty for this.
                r += 60.0 
            else:
                # Standard progression reward
                r += 15.0 + (20.0 * progress) 
            
        if info["game_over"]:
            r -= 100.0 

        if info["cells_remaining"] == 0:
            r += 500.0 + (info["steps_remaining"] * 0.5)

        return float(r)
    
    elif REWARD_STRUCTURE == "R2": # second reward structure type
        global CURRENT_DANGER_LEVEL

        if CURRENT_DANGER_LEVEL > 0:
            r = -0.1  # in a danger zone slow down
        else:
            r = -0.6  # not in danger speed up

        if info["new_cell_covered"]:
            cells_done = info["coverable_cells"] - info["cells_remaining"]
            progress = cells_done / max(info["coverable_cells"], 1)
            
            # The base reward (scales from 10 to 25)
            base_reward = 10.0 + (15.0 * progress)  
            
            r += base_reward

        if info["game_over"]:
            r -= 50.0 + (info["cells_remaining"] * 1.5)

        if info["cells_remaining"] == 0:
            r += 1000.0 + (info["steps_remaining"] * 1.0)

        return float(r)

    else:  # speed
        r = -1.0
        if info["new_cell_covered"]:
            r += 5.0
        if info["cells_remaining"] == 0:
            r += 300.0
        
        return float(r)
