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

# MONTECARLO Reinforce
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "MountainCarContinuous-v0",
    "METHOD": "ContinuousMonteCarloReinforce",
    "LOGGER_METHOD": "ContinuousPGLogger",
    "NETWORK_TYPE": "SimpleContinuousPolicyNetwork",
    "REPORTING_INTERVAL": 10,
    "LOG_LEVEL": 5,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 90, # goal is 90
    "NUM_TRIALS": 10,
    "MAX_EPISODES": 2000,
    "LEARNING_RATE": 1e-3,
    "GAMMA": 0.99,
    "BASELINE_QVALUE": True,
    "ENTROPY_BONUS": True,
    "ENTROPY_BETA": 0.01
}

# params["ENV_NAME"] = "LunarLanderContinuous-v2"
# params["MEAN_REWARD_BOUND"] = 200

# BATCH
params["METHOD"] = "ContinuousBatchReinforce"
params["EPISODE_BUFFER_SIZE"]  = 16
params["LEARNING_RATE"] = 1e-2

exp = UntilWinExperiment(params)
result = exp.run()
print("Method {} took an average of {:.2f} episodes".format(params["METHOD"], result))

# ContinuousMonteCarloReinforce
# not converging MountainCarContinuous
# not converging LunarLanderv2

# ContinuousBatchReinforce
# not converging LunarLanderv2
# not converging MountainCarContinuous