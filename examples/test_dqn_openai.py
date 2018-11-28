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
    "PLATFORM": "openai",
    "ENV_NAME": "CartPole-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "SimpleValueNetwork",
    "REPORTING_INTERVAL": 10,
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

new_params = {
    "REPORTING_INTERVAL": 10,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,    
    "MEAN_REWARD_BOUND": 195,
    "EPSILON_DECAY_LAST_FRAME": 10000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0.02,
}
params.update(new_params)

exp = UntilWinExperiment(params)
result = exp.run()
print("Method {} took an average of {:.2f} episodes".format(params["METHOD"], result))

# if is learning for mountain car, then I can check discretization
# it seems to learn better with a lot of exploration
# as soon as it starts exploring, its rewards starts to diminish
# keep constant exploration, see if it helps
# it takes at least 25x as longer as the discretized version

# solved in 405 episodes
# took 45 seconds

# solved in 150 episodes
# took 9.28 seconds

# solved in 330 episodes
# took 42.41 seconds