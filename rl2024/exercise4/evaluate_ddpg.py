import gymnasium as gym
from gymnasium import Space
from typing import List, Tuple, Dict
import numpy as np

from rl2024.exercise4.agents import DDPG
from rl2024.exercise4.train_ddpg import RACETRACK_CONFIG, play_episode


RENDER = False

CONFIG = RACETRACK_CONFIG


def evaluate(env: gym.Env, config: Dict, output: bool = True) -> Tuple[List[float], List[float]]:
    """
    Execute training of DDPG on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    obs, _ = env.reset()
    obs = obs.ravel()
    observation_space = Space((obs.shape[0],))

    agent = DDPG(
        action_space=env.action_space, observation_space=observation_space, **config
    )
    try:
        agent.restore(config['save_filename'])
    except:
        raise ValueError(f"Could not find model to load at {config['save_filename']}")

    eval_returns_all = []
    eval_times_all = []

    for loop in range(3):
        eval_returns = 0
        for _ in range(config["eval_episodes"]):
            _, episode_return, _ = play_episode(
                env,
                agent,
                0,
                train=False,
                explore=False,
                render=RENDER,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            eval_returns += episode_return / config["eval_episodes"]
        eval_returns_all.append(eval_returns)
    return eval_returns_all


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns = evaluate(env, CONFIG)
    print(returns)
    print(np.max(returns))
    env.close()
