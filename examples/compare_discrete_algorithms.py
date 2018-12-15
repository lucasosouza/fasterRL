import sys
sys.path.append("../../")

from fasterRL.common.experiment import BaseExperiment, UntilWinExperiment

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "FrozenLake-v0",
    "METHOD": "DiscreteQLearning",
    "REPORTING_INTERVAL": 100,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": .8,
    "NUM_TRIALS": 3, # compare discrete algorithms - did anything broke?
    "MAX_EPISODES": 1000,
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0,
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.99
}

params["ENV_NAME"] = "FrozenLakeNotSlippery-v0"

results = []
methods = ["QLearning", "Sarsa", "FirstVisitMonteCarlo", "EveryVisitMonteCarlo", "NStepsQLearning", "NStepsSarsa"]
for method in methods:
    params["METHOD"] = method
    exp = UntilWinExperiment(params)
    result = exp.run()
    results.append(result)

for method, result in zip(methods, results):
    print("Method {} took an average of {:.2f} episodes".format(method, result))

# Results, using all same parameters:
# For "FrozenLakeNotSlippery-v0"
# 
# Method QLearning took an average of 321.20 episodes
# Method Sarsa took an average of 500.53 episodes
# Method FirstVisitMonteCarlo took an average of 341.27 episodes
# Method EveryVisitMonteCarlo took an average of 304.73 episodes
# Method NStepsQLearning took an average of 287.47 episodes
# Method NStepsSarsa took an average of 300.37 episodes


# results in Nov-29 (3 trials only)
# nothing out of ordinary
# issue still in discretization
# Method QLearning took an average of 313.67 episodes
# Method Sarsa took an average of 348.33 episodes
# Method FirstVisitMonteCarlo took an average of 320.67 episodes
# Method EveryVisitMonteCarlo took an average of 329.67 episodes
# Method NStepsQLearning took an average of 285.33 episodes
# Method NStepsSarsa took an average of 284.33 episodes
