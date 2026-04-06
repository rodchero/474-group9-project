import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import gymnasium as gym
import coverage_gridworld  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from experiment_runner import Experiment

import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Maps (mirrors train_agent.py / test_agent.py)
# ──────────────────────────────────────────────────────────────────────────────

JUST_GO_MAP = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

SAFE_MAP = [
    [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
]

MAZE_MAP = [
    [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
]

CHOKEPOINT_MAP = [
    [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
]

TEST_MAP1 = [
    [3, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 2, 0, 2, 2, 2, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 2, 0],
    [2, 2, 2, 2, 2, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

TEST_MAP2 = [
    [3, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 2, 2, 2, 2, 2, 0],
    [0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 4, 0, 0, 0],
    [2, 2, 2, 0, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
]

TEST_MAP3 = [
    [3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 2, 0, 4, 0, 0, 2, 0],
    [2, 2, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 2, 2, 2, 0, 2, 2, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 2, 2, 0],
    [2, 2, 2, 2, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

DEFAULT_CURRICULUM = [
    ("JustGo",     JUST_GO_MAP,     150_000),
    ("Safe",       SAFE_MAP,        150_000),
    ("Maze",       MAZE_MAP,        150_000),
    ("Chokepoint", CHOKEPOINT_MAP,  150_000),
    ("JustGo",     JUST_GO_MAP,      50_000),
    ("Safe",       SAFE_MAP,         50_000),
    ("Maze",       MAZE_MAP,         50_000),
    ("Chokepoint", CHOKEPOINT_MAP,   50_000),
    ("JustGo",     JUST_GO_MAP,      50_000),
    ("Safe",       SAFE_MAP,         50_000),
    ("Maze",       MAZE_MAP,         50_000),
    ("Chokepoint", CHOKEPOINT_MAP,  100_000),
]

EVAL_MAPS = [
    ("JustGo (Train)",      JUST_GO_MAP),
    ("Safe (Train)",        SAFE_MAP),
    ("Maze (Train)",        MAZE_MAP),
    ("Chokepoint (Train)",  CHOKEPOINT_MAP),
    ("Test 1 (Easy)",       TEST_MAP1),
    ("Test 2 (Medium)",     TEST_MAP2),
    ("Test 3 (Hard)",       TEST_MAP3),
]


# ──────────────────────────────────────────────────────────────────────────────
# Worker functions (module-level so they are picklable for multiprocessing)
# ──────────────────────────────────────────────────────────────────────────────

def _train_one_agent(
    agent_idx: int,
    reward_structure: str,
    obs_structure: str,
    curriculum: list,
    cpus: int,
    checkpoint_dir: str,
    save_path: str,
) -> str:
    """
    Train a single PPO agent with the given reward/observation configuration.
    Returns the path of the saved model zip.
    """
    # Apply configuration globally in this process
    coverage_gridworld.custom.REWARD_STRUCTURE = reward_structure
    coverage_gridworld.custom.OBSERVATION_STRUCTURE = obs_structure

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50_000 // cpus),
        save_path=checkpoint_dir,
        name_prefix=f"agent_{agent_idx}",
    )

    env_kwargs = {"render_mode": None, "predefined_map": curriculum[0][1]}
    vec_env = make_vec_env("standard", n_envs=cpus, env_kwargs=env_kwargs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        tensorboard_log=None,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.94,
        ent_coef=0.06,
        policy_kwargs=dict(net_arch=[256, 128, 64]),
    )

    for i, (map_name, map_layout, steps) in enumerate(curriculum):
        print(f"  [Agent {agent_idx}] Phase {i+1}: {map_name} ({steps:,} steps)")
        print(f"[{datetime.now()}] Starting phase {i+1}: {map_name}", flush=True)
        if i > 0:
            print("Making Env")
            new_env = make_vec_env(
                "standard", n_envs=cpus,
                env_kwargs={"render_mode": None, "predefined_map": map_layout},
            )
            model.set_env(new_env)

        print("Training")
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=checkpoint_callback,
        )

    model.save(save_path)
    print(f"  [Agent {agent_idx}] Saved → {save_path}.zip")
    return save_path


def _test_one_agent(
    model_path: str,
    reward_structure: str,
    obs_structure: str,
    eval_maps: list,
    num_runs: int,
) -> dict:
    """
    Evaluate a saved PPO agent on every eval map.
    Returns {map_name: avg_coverage_pct}.
    """
    coverage_gridworld.custom.REWARD_STRUCTURE = reward_structure
    coverage_gridworld.custom.OBSERVATION_STRUCTURE = obs_structure

    model = PPO.load(model_path)
    results = {}

    for map_name, map_layout in eval_maps:
        env = gym.make("standard", render_mode=None, predefined_map=map_layout)
        pcts = []
        for _ in range(num_runs):
            obs, info = env.reset()
            done = trunc = False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, trunc, info = env.step(action)
            coverage  = info.get("total_covered_cells", 0)
            coverable = info.get("coverable_cells", 1)
            pcts.append(100.0 * coverage / coverable)
        env.close()
        results[map_name] = float(np.mean(pcts))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment subclass
# ──────────────────────────────────────────────────────────────────────────────

class RLExperiment(Experiment):
    """
    Trains N agents in parallel for a given (reward_structure, obs_structure)
    combination, evaluates each on the full map suite, then aggregates results.

    Parameters
    ----------
    reward_structure : str
        One of "R1", "R2", or "speed" — passed to custom.REWARD_STRUCTURE.
    obs_structure : str
        One of "O1" or "O2" — passed to custom.OBSERVATION_STRUCTURE.
    num_agents : int
        How many independent agents to train (default 5).
    curriculum : list[tuple] or None
        List of (name, map, steps) triples. Defaults to DEFAULT_CURRICULUM.
    eval_maps : list[tuple] or None
        List of (name, map) pairs to test on. Defaults to EVAL_MAPS.
    num_test_runs : int
        Episodes per map per agent during evaluation (default 20).
    cpus_per_agent : int
        Parallel gym environments per agent during training (default 4).
    output_folder : str
        Root folder for all run outputs (default "output").
    """

    def __init__(
        self,
        reward_structure: str = "R1",
        obs_structure: str = "O1",
        num_agents: int = 5,
        curriculum: list | None = None,
        eval_maps: list | None = None,
        num_test_runs: int = 20,
        cpus_per_agent: int = 4,
        output_folder: str = "output",
    ):
        model_name = f"{reward_structure}_{obs_structure}"
        # input_path=None disables caching (training is stochastic)
        super().__init__(
            experiment_name="RLExperiment",
            model_name=model_name,
            input_path=None,
            output_folder=output_folder,
        )

        self.reward_structure = reward_structure
        self.obs_structure    = obs_structure
        self.num_agents       = num_agents
        self.curriculum       = curriculum or DEFAULT_CURRICULUM
        self.eval_maps        = eval_maps  or EVAL_MAPS
        self.num_test_runs    = num_test_runs
        self.cpus_per_agent   = cpus_per_agent

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run_experiment(self):
        """
        1. Train `num_agents` agents in parallel.
        2. Test each agent on every eval map.
        3. Aggregate coverage results across agents.
        4. Return (results_array, figures, fig_names).
        """
        run_folder  = self.make_run_folder()
        ckpt_root   = os.path.join(run_folder, "checkpoints")
        models_root = os.path.join(run_folder, "models")
        os.makedirs(ckpt_root,   exist_ok=True)
        os.makedirs(models_root, exist_ok=True)

        map_names = [m for m, _ in self.eval_maps]

        # ── Phase 1: Train ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Training {self.num_agents} agents  "
              f"[reward={self.reward_structure}, obs={self.obs_structure}]")
        print(f"{'='*60}")

        model_paths = [
            os.path.join(models_root, f"agent_{i}")
            for i in range(self.num_agents)
        ]

        for i in range(self.num_agents):
            try:
                _train_one_agent(
                    i,
                    self.reward_structure,
                    self.obs_structure,
                    self.curriculum,
                    self.cpus_per_agent,
                    os.path.join(ckpt_root, f"agent_{i}"),
                    model_paths[i],
                )
                print(f"  Agent {i} training complete.")
            except Exception as exc:
                print(f"  Agent {i} FAILED during training: {exc}")
                raise

        # ── Phase 2: Test ─────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Testing {self.num_agents} agents  ({self.num_test_runs} runs/map)")
        print(f"{'='*60}")

        # results_matrix[agent_idx, map_idx] = avg_coverage_pct
        results_matrix = np.zeros((self.num_agents, len(map_names)))

        for i in range(self.num_agents):
            try:
                agent_results = _test_one_agent(
                    model_paths[i],
                    self.reward_structure,
                    self.obs_structure,
                    self.eval_maps,
                    self.num_test_runs,
                )
                for j, name in enumerate(map_names):
                    results_matrix[i, j] = agent_results[name]
                print(f"  Agent {i} testing complete.")
            except Exception as exc:
                print(f"  Agent {i} FAILED during testing: {exc}")
                raise

        # ── Phase 3: Summarise ────────────────────────────────────────
        mean_cov = results_matrix.mean(axis=0)   # (num_maps,)
        std_cov  = results_matrix.std(axis=0)

        print(f"\n{'='*60}")
        print(f"Results  [reward={self.reward_structure}, obs={self.obs_structure}]")
        print(f"{'='*60}")
        header = f"{'Map':<25} {'Mean %':>8} {'Std':>7}"
        print(header)
        print("-" * len(header))
        for name, m, s in zip(map_names, mean_cov, std_cov):
            print(f"{name:<25} {m:>7.2f}% {s:>6.2f}")

        # ── Phase 4: Figures ──────────────────────────────────────────
        figures, fig_names = self._make_figures(
            map_names, mean_cov, std_cov, results_matrix
        )

        return results_matrix, figures, fig_names

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _make_figures(self, map_names, mean_cov, std_cov, results_matrix):
        figures, fig_names = [], []

        # Figure 1 — bar chart: mean ± std per map
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        x = np.arange(len(map_names))
        bars = ax1.bar(x, mean_cov, yerr=std_cov, capsize=5,
                       color="steelblue", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(map_names, rotation=30, ha="right", fontsize=9)
        ax1.set_ylabel("Average Coverage (%)")
        ax1.set_ylim(0, 110)
        ax1.set_title(
            f"Coverage by Map  "
            f"[reward={self.reward_structure}, obs={self.obs_structure}, "
            f"N={self.num_agents} agents]"
        )
        ax1.axvline(3.5, color="grey", linestyle="--", linewidth=0.8,
                    label="train / test split")
        ax1.legend()
        for bar, m in zip(bars, mean_cov):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{m:.1f}", ha="center", va="bottom", fontsize=8)
        fig1.tight_layout()
        figures.append(fig1)
        fig_names.append("coverage_by_map")

        # Figure 2 — per-agent coverage heatmap
        fig2, ax2 = plt.subplots(figsize=(12, max(3, self.num_agents * 0.6 + 1)))
        im = ax2.imshow(results_matrix, aspect="auto", vmin=0, vmax=100,
                        cmap="RdYlGn")
        ax2.set_xticks(np.arange(len(map_names)))
        ax2.set_xticklabels(map_names, rotation=30, ha="right", fontsize=9)
        ax2.set_yticks(np.arange(self.num_agents))
        ax2.set_yticklabels([f"Agent {i}" for i in range(self.num_agents)])
        ax2.set_title(
            f"Per-Agent Coverage Heatmap  "
            f"[reward={self.reward_structure}, obs={self.obs_structure}]"
        )
        for i in range(self.num_agents):
            for j in range(len(map_names)):
                ax2.text(j, i, f"{results_matrix[i, j]:.0f}",
                         ha="center", va="center", fontsize=8,
                         color="black")
        plt.colorbar(im, ax=ax2, label="Coverage (%)")
        fig2.tight_layout()
        figures.append(fig2)
        fig_names.append("agent_heatmap")

        return figures, fig_names


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    reward_structures = ["R1", "R2", "speed"]
    obs_structures    = ["O1", "O2"]
    EXCLUDE           = {("O1", "R2")}   # (obs, reward) pairs to skip

    combinations = [
        (r, o)
        for r in reward_structures
        for o in obs_structures
        if (o, r) not in EXCLUDE
    ]

    NUM_AGENTS = 5
    TOTAL_CPUS = os.cpu_count()

    print(f"Running {len(combinations)} combinations sequentially "
          f"({NUM_AGENTS} agents each):")
    for r, o in combinations:
        print(f"  * {r}+{o}")
    print()
   

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", type=int, help="id of the experiment to run")
    args = parser.parse_args()

    print(f"Running {len(combinations)} combinations sequentially "
          f"({NUM_AGENTS} agents in parallel each):")
    for r, o in combinations:
        print(f"  * {r}+{o}")
    print()

    for i in range(5):
        #i = args.experiment_id + 1
        r, o = combinations[args.experiment_id]

        label = f"{r}+{o}"
        print("\n" + "=" * 60)
        print(f"Combination {i}/{len(combinations)}: {label}")
        print("=" * 60)
        exp = RLExperiment(
            reward_structure=r,
            obs_structure=o,
            num_agents=NUM_AGENTS,
            num_test_runs=20,
            cpus_per_agent=TOTAL_CPUS,
            output_folder="output",
        )
        exp()
        print(f"{label} complete")