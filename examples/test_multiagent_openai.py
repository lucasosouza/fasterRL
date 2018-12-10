import sys
sys.path.append("../../")

from fasterRL.common.experiment import MultiAgentExperiment

params = {
    "PLATFORM": "openai",
    "ENV_NAME": "FrozenLakeNotSlippery-v0",
    "METHOD": "QLearning",
    "LOGGER_METHOD": "WinLogger",
    "REPORTING_INTERVAL": 100,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": .8,
    "NUM_TRIALS": 10,
    "MAX_EPISODES": 1000,
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0,
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.99,
    "NUM_AGENTS": 2
}

exp = MultiAgentExperiment(params)
result = exp.run()


    
