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

params = {
    "PLATFORM": "openai-atari",
    "ENV_NAME": "Pong-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "DeepQNetwork",
    "REPORTING_INTERVAL": 1,
    "LOG_LEVEL": 2, #debugging level
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 199,
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

# params["PLATFORM"] = "openai"
# params["ENV_NAME"] = "CartPole-v0"
# params["NETWORK_TYPE"] = "SimpleValueNetwork"

exp = UntilWinExperiment(params)
result = exp.run()
print("Method {} took an average of {:.2f} episodes".format(params["METHOD"], result))

# Method DQN took an average of 263.90 episodes

# will be too slow to run atari 
# want to at least see if it is running - should be able to run more than one episode at least
# that should suffice for now

# it is running:
# Episode 1 Mean Reward: -21.000 Mean Steps: 1133.000 Speed: 172.345 sec/ep Frame: 1133
# Episode 2 Mean Reward: -20.000 Mean Steps: 1313.000 Speed: 234.758 sec/ep Frame: 2446
# Episode 3 Mean Reward: -21.000 Mean Steps: 1066.000 Speed: 190.153 sec/ep Frame: 3512

# but it does not mean it is learning

# need to adapt focused sharing buffer to handle images as well
# seems to be working 
