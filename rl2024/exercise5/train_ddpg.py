import copy
import pickle
from collections import defaultdict

import gymnasium as gym
from gymnasium import Space
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from rl2024.constants import EX5_RACETRACK_CONSTANTS as RACETRACK_CONSTANTS

from rl2024.exercise4.agents import DDPG
from rl2024.exercise4.train_ddpg import train
from rl2024.exercise3.replay import ReplayBuffer
from rl2024.util.hparam_sweeping import generate_hparam_configs
from rl2024.util.result_processing import Run

RENDER = False
SWEEP = True # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 3 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGTHS = True # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "RACETRACK"

# IN EXERCISE 5 YOU SHOULD TUNE PARAMETERS IN THIS CONFIG ONLY
RACETRACK_CONFIG = {
    "policy_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [32, 32, 32],
    "policy_hidden_size": [32, 32, 32],
    "gamma": 0.99,
    "tau": 0.5,
    "batch_size": 32,
    "buffer_capacity": int(1e6),
}
RACETRACK_CONFIG.update(RACETRACK_CONSTANTS)

### INCLUDE YOUR CHOICE OF HYPERPARAMETERS HERE ###
RACETRACK_HPARAMS = {
    "critic_hidden_size": ...,
    "policy_hidden_size": ...,
}

SWEEP_RESULTS_FILE_BIPEDAL = "DDPG-Racetrack-sweep-results-ex5.pkl"

if __name__ == "__main__":
    if ENV == "RACETRACK":
        CONFIG = RACETRACK_CONFIG
        HPARAMS_SWEEP = RACETRACK_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_BIPEDAL
    else:
        raise (ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])
    env_eval = gym.make(CONFIG["env"])

    if SWEEP and HPARAMS_SWEEP is not None:
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
        results = []
        for config in config_list:
            run = Run(config)
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            run.run_name = hparams_values
            print(f"\nStarting new run...")
            for i in range(NUM_SEEDS_SWEEP):
                print(f"\nTraining iteration: {i + 1}/{NUM_SEEDS_SWEEP}")
                run_save_filename = '--'.join([run.config["algo"], run.config["env"], hparams_values, str(i)])
                if SWEEP_SAVE_ALL_WEIGTHS:
                    run.set_save_filename(run_save_filename)
                eval_returns, eval_timesteps, times, run_data = train(env, env_eval, run.config, output=True)
                run.update(eval_returns, eval_timesteps, times, run_data)
            results.append(copy.deepcopy(run))
            print(f"Finished run with hyperparameters {hparams_values}. "
                  f"Mean final score: {run.final_return_mean} +- {run.final_return_ste}")

        if SWEEP_SAVE_RESULTS:
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)

    else:
        raise NotImplementedError('You are attempting to run normal training within the hyperparameter tuning file!')

    env.close()
