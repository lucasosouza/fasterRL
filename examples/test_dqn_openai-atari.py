


from fasterrl.common.experiment import UntilWinExperiment

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

