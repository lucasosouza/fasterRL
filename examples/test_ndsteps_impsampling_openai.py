import sys
sys.path.append("../../")

from fasterRL.common import BaseExperiment, UntilWinExperiment

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "FrozenLake-v0",
    "METHOD": "DiscreteQLearning",
    "REPORTING_INTERVAL": 100,
    "LOG_LEVEL": 1,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": .8,
    "NUM_TRIALS": 30,
    "MAX_EPISODES": 1000,
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0.001, # can't be zero, since behavior policy needs to "cover" target policy
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.99,
    "IMPORTANCE_SAMPLING": True,
    "SAMPLING_PER_DECISION": True,
    "N_STEPS": 3
}

params["ENV_NAME"] = "FrozenLakeNotSlippery-v0"

methods = ["NStepsQLearning", "NStepsSarsa"]
n_steps = [5]
imp_samplings = [True]
# n_steps = [3,5,10]
# imp_samplings = [True, False]

results = []
experiments = []
for method in methods:
    for n_step in n_steps:
        for imp_sampling in imp_samplings:
            params["METHOD"] = method
            params["N_STEPS"] = n_step
            params["IMPORTANCE_SAMPLING"] = imp_sampling

            exp = UntilWinExperiment(params)
            result = exp.run()
            results.append(result)
            experiments.append("{}-{}-{}".format(method, n_step, imp_sampling))

for method, result in zip(experiments, results):
    print("Method {} took an average of {:.2f} episodes".format(method, result))

# Results, using all same parameters:
# For "FrozenLakeNotSlippery-v0"

# Method NStepsQLearning-3-True took an average of 305.63 episodes
# Method NStepsQLearning-3-False took an average of 274.23 episodes
# Method NStepsQLearning-5-True took an average of 290.20 episodes
# Method NStepsQLearning-5-False took an average of 275.70 episodes
# Method NStepsQLearning-10-True took an average of 309.17 episodes
# Method NStepsQLearning-10-False took an average of 297.23 episodes

# Method NStepsSarsa-3-True took an average of 310.23 episodes
# Method NStepsSarsa-3-False took an average of 309.40 episodes
# Method NStepsSarsa-5-True took an average of 315.57 episodes
# Method NStepsSarsa-5-False took an average of 288.33 episodes
# Method NStepsSarsa-10-True took an average of 322.00 episodes
# Method NStepsSarsa-10-False took an average of 300.90 episodes

# summary all methods with importance sampling took a bit longer to converge in this environment.

# with importance sampling
# Method NStepsQLearning-5-True took an average of 278.00 episodes
# Method NStepsSarsa-5-True took an average of 299.50 episodes

# wuth per decision
# Method NStepsQLearning-5-True took an average of 754.27 episodes
# Method NStepsSarsa-5-True took an average of 776.73 episodes
