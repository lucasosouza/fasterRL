


from fasterrl.common import BaseExperiment, UntilWinExperiment

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "FrozenLake-v0",
    "METHOD": "DiscreteQLearning",
    "REPORTING_INTERVAL": 100,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": .8,
    "NUM_TRIALS": 30,
    "MAX_EPISODES": 1000,
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0.001, # can't be zero, since behavior policy needs to "cover" target policy
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.99,
    "IMPORTANCE_SAMPLING": True
}

params["ENV_NAME"] = "FrozenLakeNotSlippery-v0"

results = []
methods = ["FirstVisitMonteCarlo", "EveryVisitMonteCarlo"]
for method in methods:
    params["METHOD"] = method
    exp = UntilWinExperiment(params)
    result = exp.run()
    results.append(result)

for method, result in zip(methods, results):
    print("Method {} took an average of {:.2f} episodes".format(method, result))

# Results, using all same parameters:
# For "FrozenLakeNotSlippery-v0"

# Method FirstVisitMonteCarlo took an average of 425.40 episodes
# Method EveryVisitMonteCarlo took an average of 385.47 episodes

# compared to non importance sampling
# Method FirstVisitMonteCarlo took an average of 341.27 episodes
# Method EveryVisitMonteCarlo took an average of 304.73 episodes
