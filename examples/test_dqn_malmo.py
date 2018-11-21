import sys
sys.path.append("../../")

from fasterRL.common.experiment import UntilWinExperiment

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
    "PLATFORM": "malmo",
    "ENV_NAME": "MinecraftBasicNew-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "DeepQNetwork",
    "REPORTING_INTERVAL": 1,
    "RENDER": True,
    "LOG_LEVEL": 2, #debugging level
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 950,
    "NUM_TRIALS": 1,
    "MAX_EPISODES": 3000,
    "EPSILON_DECAY_LAST_FRAME": 4000, # 4000
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0,
    "LEARNING_RATE": 1e-3,
    "GAMMA": 0.99,
    "REPLAY_BATCH_SIZE": 32,
    "EXPERIENCE_BUFFER_SIZE": 5000,
    "GRADIENT_CLIPPING": False,
    "DOUBLE_QLEARNING": True,
    "SOFT_UPDATE": True, 
    "SOFT_UPDATE_TAU": 5e-3 
}

exp = UntilWinExperiment(params)
result = exp.run()
print("Method {} took an average of {:.2f} episodes".format(params["METHOD"], result))

# works
# with an average speed of 4 seconds per episode
# that seems a lot, not sure how it compares with previous versions of the library
# see if I can speed up
# also no immediate signs of learning - however, since DQN has already been tested, and it is just a matter of adapting the algorithm, for now we can trust its learning and conduct tests later