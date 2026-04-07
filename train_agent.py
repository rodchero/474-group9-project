import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from datetime import datetime

#CPU cores used for training
CPUS = 8

# our maps
just_go_map = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

safe_map = [
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

maze_map = [
    [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
]

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

# training ciricullum
curriculum = [
    ("JustGo",     just_go_map,     150_000),
    ("Safe",       safe_map,        150_000),
    ("Maze",       maze_map,        150_000),
    ("Chokepoint", chokepoint_map,  150_000),
    ("JustGo",     just_go_map,     50_000),
    ("Safe",       safe_map,        50_000),
    ("Maze",       maze_map,        50_000),
    ("Chokepoint", chokepoint_map,  50_000),
    ("JustGo",     just_go_map,     50_000),
    ("Safe",       safe_map,        50_000),
    ("Maze",       maze_map,        50_000),
    ("Chokepoint", chokepoint_map,  100_000),
]
total_training = sum(s for _, _, s in curriculum)
print(f"Total timesteps: {total_training}")


def run_train(save_num, reward_structure, obs_structure):
    coverage_gridworld.custom.REWARD_STRUCTURE = reward_structure
    coverage_gridworld.custom.OBSERVATION_STRUCTURE = obs_structure

    # model saving to ensure we keep the best model before it collapses
    checkpoint_callback = CheckpointCallback(
        save_freq=12500, 
        save_path='./ppo_checkpoints/',
        name_prefix='ppo_agent'
    )

    # environment parallelization 
    env_kwargs = {"render_mode": None, "predefined_map": curriculum[0][1]}
    first_vec_env = make_vec_env("standard", n_envs=CPUS, env_kwargs=env_kwargs)

    # PPO model defined HERE
    model = PPO(
        "MlpPolicy",
        first_vec_env,
        tensorboard_log=f"./ppo_tensorboard_logs/PPO_e2_{save_num}",
        verbose=2,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.94,
        ent_coef=0.06,  
        policy_kwargs=dict(net_arch=[256, 128, 64]) # they call me "the funnel"
    )


    # train the model with our ciricullum
    for i, (map_name, map_layout, steps) in enumerate(curriculum):
        print(f"[{datetime.now()}] Starting phase {i+1}: {map_name}", flush=True)

        if i > 0:
            # creating new parallel enviroment for next map
            new_env_kwargs = {"render_mode": None, "predefined_map": map_layout}
            new_vec_env = make_vec_env("standard", n_envs=CPUS, env_kwargs=new_env_kwargs)
            model.set_env(new_vec_env)

        # callback added here
        model.learn(
            total_timesteps=steps, 
            reset_num_timesteps=False, 
            callback=checkpoint_callback
        )

    # model saving
    save_name = f"ppo_agent{save_num}"
    model.save(save_name)
    print(f"\nTraining Complete! Saved as {save_name}")
    
    env_human.close()


reward_structures = ["R1", "R2", "speed"]
obs_structures    = ["O1", "O2"]
EXCLUDE           = {("O1", "R2")}   # (obs, reward) pairs to skip

combinations = [
    (r, o)
    for r in reward_structures
    for o in obs_structures
    if (o, r) not in EXCLUDE
]

for i, (r, o) in enumerate(combinations):
    run_train(i,r,o)