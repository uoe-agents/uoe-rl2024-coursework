import gymnasium as gym
from typing import List, Tuple

from rl2024.exercise4.agents import DDPG
from rl2024.exercise4.evaluate_ddpg import evaluate
from rl2024.exercise5.train_ddpg \
    import RACETRACK_CONFIG

RENDER = False

CONFIG = RACETRACK_CONFIG

if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
