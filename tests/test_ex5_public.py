"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import gymnasium as gym
from gymnasium import Space
import os.path
import numpy as np

def test_imports():
    from rl2024.exercise4 import DDPG
    from rl2024.exercise5.train_ddpg import RACETRACK_CONFIG as CONFIG

def test_config():
    from rl2024.exercise5.train_ddpg import RACETRACK_CONFIG
    assert "episode_length" in RACETRACK_CONFIG
    assert "max_timesteps" in RACETRACK_CONFIG
    assert "eval_freq" in RACETRACK_CONFIG
    assert "eval_episodes" in RACETRACK_CONFIG
    assert "policy_learning_rate" in RACETRACK_CONFIG
    assert "critic_learning_rate" in RACETRACK_CONFIG
    assert "policy_hidden_size" in RACETRACK_CONFIG
    assert "critic_hidden_size" in RACETRACK_CONFIG
    assert "tau" in RACETRACK_CONFIG
    assert "batch_size" in RACETRACK_CONFIG
    assert "gamma" in RACETRACK_CONFIG
    assert "buffer_capacity" in RACETRACK_CONFIG
    assert "save_filename" in RACETRACK_CONFIG


def test_restore_file():
    from rl2024.exercise4 import DDPG
    from rl2024.exercise5.train_ddpg import RACETRACK_CONFIG
    env = gym.make("racetrack-v0")
    obs, _ = env.reset()
    obs = obs.ravel()
    observation_space = Space((obs.shape[0],))
    agent = DDPG(
        action_space=env.action_space,
        observation_space=observation_space,
        **RACETRACK_CONFIG
    )
    save_dir, _ = os.path.split(os.path.abspath(__file__))
    save_dir, _ = os.path.split(save_dir)
    save_dir = os.path.join(save_dir, 'rl2024/exercise5')
    agent.restore("racetrack_hparam_latest.pt", dir_path=save_dir)
