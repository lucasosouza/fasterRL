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
    "ENV_NAME": "CartPole-v0",
    "METHOD": "A2C",
    "LOGGER_METHOD": "A2CLogger",
    "NETWORK_TYPE": "SimpleA2CNetwork",
    "REPORTING_INTERVAL": 10,
    "LOG_LEVEL": 5,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 199,
    "NUM_TRIALS": 1,
    "MAX_EPISODES": 10000,
    "LEARNING_RATE": 2e-4,
    "GAMMA": 0.99,
    "ENTROPY_BONUS": True, 
    "ENTROPY_BETA": 0.01,
    "GRADIENT_CLIPPING": True,
    "CLIP_GRAD": 0.1
}

exp = UntilWinExperiment(params)
result = exp.run()
print("Method {} took an average of {:.2f} episodes".format(params["METHOD"], result))
