EX1_CONSTANTS = {
    "gamma": 0.85,
}

EX2_CONSTANTS = {
    "env": "FrozenLake8x8-v1",
    "eps_max_steps": 200,
    "eval_episodes": 500,
    "eval_eps_max_steps": 200,
}

EX2_MC_CONSTANTS = EX2_CONSTANTS.copy()
EX2_MC_CONSTANTS["total_eps"] = 300000

EX2_QL_CONSTANTS = EX2_CONSTANTS.copy()
EX2_QL_CONSTANTS["total_eps"] = 10000

EX3_CARTPOLE_CONSTANTS = {
    "env": "CartPole-v0",
    "gamma": 0.99,
    "episode_length": 200,
    "max_time": 30 * 60,
    "save_filename": None,
    "algo": None,
}

EX3_DQN_CARTPOLE_CONSTANTS = EX3_CARTPOLE_CONSTANTS.copy()
EX3_DQN_CARTPOLE_CONSTANTS["max_timesteps"] = 40000
EX3_DQN_CARTPOLE_CONSTANTS["algo"] = "DQN"

EX3_REINFORCE_CARTPOLE_CONSTANTS = EX3_CARTPOLE_CONSTANTS.copy()
EX3_REINFORCE_CARTPOLE_CONSTANTS["max_timesteps"] = 500000
EX3_REINFORCE_CARTPOLE_CONSTANTS["algo"] = "Reinforce"

EX3_MOUNTAINCAR_CONSTANTS = {
    "env": "MountainCar-v0",
    "gamma": 0.99,
    "episode_length": 200,
    "max_time": 120 * 60,
    "save_filename": None,
    "algo": None,
}

EX3_DQN_MOUNTAINCAR_CONSTANTS = EX3_MOUNTAINCAR_CONSTANTS.copy()
EX3_DQN_MOUNTAINCAR_CONSTANTS["max_timesteps"] = 700000
EX3_DQN_MOUNTAINCAR_CONSTANTS["algo"] = "DQN"

EX4_RACETRACK_CONSTANTS = {
    "env": "racetrack-v0",
    "target_return": 500.0,
    "episode_length": 31000,
    "max_timesteps": 200,
    "max_time": 120 * 60,
    "gamma": 0.95,
    "save_filename": "racetrack_latest.pt",
    "eval_freq": 100,
    "eval_episodes": 5,
    "policy_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "tau": 0.005,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
    "algo": "DDPG",
}

EX5_RACETRACK_CONSTANTS = {
    "env": "racetrack-v0",
    "target_return": 800.0,
    "episode_length": 300,
    "max_timesteps": 31000,
    "max_time": 120 * 60,
    "gamma": 0.99,
    "save_filename": "racetrack_hparam_latest.pt",
    "eval_freq": 3000,
    "eval_episodes": 5,
    "policy_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "tau": 0.005,
    "batch_size": 32,
    "buffer_capacity": int(1e6),
    "algo": "DDPG",
}
