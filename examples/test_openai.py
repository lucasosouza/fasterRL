import sys
sys.path.append("../../")

from fasterRL.common import BaseExperiment, UntilWinExperiment

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "FrozenLake-v0",
#    "ENV_NAME": "FrozenLakeNotSlippery-v0",
    "METHOD": "DiscreteQLearning",
    "NUM_TRIALS": 10,
    "NUM_EPISODES": 1000,
    "REPORTING_INTERVAL": 100,
    "LOG_LEVEL": 4,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": .8,
    "MAX_EPISODES": 1000,
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0,
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.995
}

exp = UntilWinExperiment(params)
exp.run()
