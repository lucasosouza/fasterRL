

from fasterrl.common.experiment import UntilWinExperiment

params = {
    "LOG_LEVEL": 2,
    "PLATFORM": "openai",
    "ENV_NAME": "FrozenLake-v0",
    "METHOD": "QLearning",
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": .8,
    "REPORTING_INTERVAL": 100,
    "NUM_TRIALS": 3,
    "MAX_EPISODES": 1000,
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.99
}

exp = UntilWinExperiment(params)
exp.run()