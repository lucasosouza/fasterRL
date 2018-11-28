import sys
sys.path.append("../../")

from fasterRL.common.experiment import UntilWinExperiment

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "MountainCarContinuous-v0",
    "METHOD": "QLearning",
    "REPORTING_INTERVAL": 100,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 90,
    "NUM_TRIALS": 1,
    "MAX_EPISODES": 10000,
    "EPSILON_DECAY_LAST_FRAME": 1000000, # 500000 do not solve
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0.02, # 0.02
    "LEARNING_RATE": 0.05,
    "GAMMA": 0.99,
    "DISCRETIZE_STATE": True,
    "DISCRETIZE_STATE_BIN_SIZE": 10,
    "DISCRETIZE_ACTION": True,
    "DISCRETIZE_ACTION_BIN_SIZE": 50,
}

exp = UntilWinExperiment(params)
exp.run()

# solved the problem using action discretization 
# Problem solved in 1839 episodes
# Trial took 62.19 seconds

