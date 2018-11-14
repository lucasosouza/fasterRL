import sys
sys.path.append("../../")

from fasterRL.common import BaseExperiment, UntilWinExperiment

"""
    LOG LEVELS:
    1 - report nothing, just run
    2 - print to screen
    3 - log episode-wise variables
    4 - log step-wise variable
    5 - log specifics relevant for debugging
"""

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "CartPole-v0",
    "METHOD": "CrossEntropy",
    "LOGGER_METHOD": "CrossEntropyLogger",
    "NETWORK_TYPE": "SimplePolicyNetwork",
    "REPORTING_INTERVAL": 10,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 199,
    "NUM_TRIALS": 10,
    "MAX_EPISODES": 3000,
    "LEARNING_RATE": 1e-2,
    "GAMMA": 0.99,
    "EPISODE_BUFFER_SIZE": 16,
    "CUTOFF_PERCENTILE": 70
}

exp = UntilWinExperiment(params)
result = exp.run()
print("Method {} took an average of {:.2f} episodes".format(params["METHOD"], result))

# Method CrossEntropy took an average of 597.60 episodes