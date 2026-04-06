"""
This file contains helpers for running experiments.
"""


import time
import os
import sys
import numpy as np
from datetime import datetime

import logging
    
from concurrent.futures import ThreadPoolExecutor, as_completed


class Experiment:
    """Wrapper for generic code that handles saving and retrieving experiment data"""

    def __init__(self, experiment_name, model_name, input_path, output_folder="output"):
        self.OUTPUT_FOLDER   = output_folder
        self.TIMESTAMP       = time.time()
        self.EXPERIMENT_NAME = experiment_name
        self.MODEL_NAME      = model_name
        if input_path:
            self.input_path      = input_path + "_" + model_name


    # ======================================================================
    # Abstract methods for generating inputs and running the experiments
    # ======================================================================

    def generate_input(self):
        pass

    def run_experiment(self):
        pass


    # ======================================================================
    # Saving and caching data
    # ======================================================================

    def make_run_folder(self):
        """
        Creates and returns a timestamped folder for this run's outputs.
        Format sorts cleanly in file explorer by most recent:

            <output_folder>/<experiment_name>/YYYY-MM-DD_HH-MM-SS_<model_name>/
        """
        timestamp_str = datetime.fromtimestamp(self.TIMESTAMP).strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join(self.OUTPUT_FOLDER, self.EXPERIMENT_NAME, f"{timestamp_str}_{self.MODEL_NAME}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def save_output(self, results, figures, fig_names=None):
        """
        Save results array and all figures into the run folder.

        Args:
            results:   np.ndarray of experiment results.
            figures:   List of matplotlib figures to save.
            fig_names: Optional list of names for each figure. Defaults to fig_1, fig_2, ...
        """
        if fig_names is not None and len(fig_names) != len(figures):
            raise ValueError(f"fig_names length {len(fig_names)} doesn't match figures length {len(figures)}")

        folder = self.make_run_folder()

        # Save results array
        results_path = os.path.join(folder, "results.npy")
        np.save(results_path, results)
        print(f"Results saved to {results_path}")

        # Save figures
        for i, fig in enumerate(figures):
            name = fig_names[i] if fig_names else f"fig_{i+1}"
            fig_path = os.path.join(folder, f"{name}.pdf")
            fig.savefig(fig_path)
            print(f"Figure saved to {fig_path}")

        return folder

    def collect_data(self, expected_shape, *args, **kwargs):
        """
        Load from cache if valid, otherwise call generate_input and save the result.

        Args:
            expected_shape: Tuple of the expected shape of the result array. Used to validate the cache.
            *args:          Positional arguments forwarded to generate_input.
            **kwargs:       Keyword arguments forwarded to generate_input.

        Returns:
            np.ndarray of shape expected_shape.
        """
        if os.path.exists(self.input_path):
            print(f"Loading cached results from {self.input_path}")
            data = np.load(self.input_path)

            if data.shape == expected_shape:
                print("Cache valid, using cached data")
                return data

            print(f"Cache shape {data.shape} doesn't match expected {expected_shape}, recomputing")

        data = self.generate_input(*args, **kwargs)
        np.save(self.input_path, data)
        print(f"Saved input to {self.input_path}")
        return data

    def __call__(self, *args, **kwargs):
        print(f"Running experiment {self.EXPERIMENT_NAME} using model {self.MODEL_NAME}")
        output    = self.run_experiment(*args, **kwargs)
        results, figures, names = output
        folder    = self.save_output(results, figures, names)

        return results