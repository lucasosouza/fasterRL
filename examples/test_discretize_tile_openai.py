import sys
sys.path.append("../../")

from fasterRL.common.experiment import UntilWinExperiment
import numpy as np

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "CartPole-v0",
    "METHOD": "QLearning",
    "REPORTING_INTERVAL": 100,
    "LOG_LEVEL": 2,
    "LOGGER_METHOD": "TDLogger",
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 195,
    "NUM_TRIALS": 5,
    "MAX_EPISODES": 10000,
    "EPSILON_DECAY_LAST_FRAME": 50000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0.02, # 0.02
    "LEARNING_RATE": 0.05,
    "GAMMA": 0.99,
    "DISCRETIZE_STATE": True,
    "DISCRETIZE_BIN_SIZE": 10,
    "DISCRETIZE_TYPE": 'unitary',
   "WITH_TILES": True,
#    "TILE_OFFSETS": [0, -0.05, 0.05, -0.1, 0.1]
    "TILE_OFFSETS": list(np.arange(-0.10, 0.11, 0.01))
}


# was working - maybe something changed?

exp = UntilWinExperiment(params)
exp.run()

# with TILES default : Average number of episodes for trial: 961.20

# with TILES range(-0.10, 0.12, 0.02): Average number of episodes for trial: 792.20
# with TILES np.arange(-0.20, 0.22, 0.02): Average number of episodes for trial: 784.60
# with TILES range(-0.10, 0.11, 0.01): Average number of episodes for trial: 837.60

# significant improvement after the first
# but not a lot of improvement after the first range with 11 tiles total (default + 5 each side)

# without TILES: Average number of episodes for trial: 4227.00


