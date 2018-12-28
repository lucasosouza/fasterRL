import sys
sys.path.append("../../")

from fasterRL.common.experiment import BaseExperiment, UntilWinExperiment

params = {
    "PLATFORM": "openai",
    "METHOD": "DDPG",
    "LOGGER_METHOD": "WinLogger",
    "REPORTING_INTERVAL": 10,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "NUM_TRIALS": 1,
    "MAX_EPISODES": 2000,
    "LEARNING_RATE": 2e-4,
    "GAMMA": 0.99,
    "SOFT_UPDATE": True, 
    "SOFT_UPDATE_TAU": 5e-3,    
    "OU_EXPLORATION": True,
    "OU_MU": 0.0,
    "OU_THETA": 0.15,
    "OU_SIGMA": 0.2,
    "OU_EPSILON": 1.0,
    # "DEVICE": 'cuda'    
}

# test for mountain car
params["ENV_NAME"] = "MountainCarContinuous-v0"
params["MEAN_REWARD_BOUND"] = 90

# test for lunar lander
# params["ENV_NAME"] = "LunarLanderContinuous-v2"
# params["MEAN_REWARD_BOUND"] = 200

# BATCH
exp = UntilWinExperiment(params)
result = exp.run()

# LunarLanderContinuous-v2
# Method DDPG took an average of 456.00 episodes

# MountainCarContinuous-v0
# Method DDPG took an average of 41.00 episodes

# ddpg can actually take a lot longer than I imagine ....
