import sys
sys.path.append("../../")

from fasterRL.common import BaseExperiment, UntilWinExperiment

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "FrozenLake-v0",
    "METHOD": "DiscreteQLearning",
    "REPORTING_INTERVAL": 1000,
    "LOG_LEVEL": 1,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": .8,
    "NUM_TRIALS": 30,
    "MAX_EPISODES": 1000,
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0,
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.99
}

params["ENV_NAME"] = "FrozenLakeNotSlippery-v0"

results = []
methods = ["QLearning", "Sarsa", "FirstVisitMonteCarlo", "EveryVisitMonteCarlo"]
for method in methods:
    params["METHOD"] = method
    exp = UntilWinExperiment(params)
    result = exp.run()
    results.append(result)

for method, result in zip(methods, results):
    print("Method {} took an average of {:.2f} episodes".format(method, result))

# Results:
# Method QLearning took an average of 323.87 episodes
# Method Sarsa took an average of 403.97 episodes
# Method FirstVisitMonteCarlo took an average of 328.93 episodes
# Method EveryVisitMonteCarlo took an average of 313.17 episodes
