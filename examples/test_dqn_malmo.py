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
    "PLATFORM": "malmo",
    "ENV_NAME": "MinecraftBasicNew-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "DeepQNetwork",
    "REPORTING_INTERVAL": 1,
    "RENDER": True,
    "LOG_LEVEL": 2, #debugging level
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 980,
    "NUM_TRIALS": 5,
    "MAX_EPISODES": 100,
    "EPSILON_DECAY_LAST_FRAME": 2000, # 4000
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0.01, # 0.02
    "LEARNING_RATE": 1e-4, # 1e-3
    "GAMMA": 0.97, # 0.99
    "REPLAY_BATCH_SIZE": 128, # 32
    "EXPERIENCE_BUFFER_SIZE": 10000, # 5000
    "GRADIENT_CLIPPING": False,
    "DOUBLE_QLEARNING": True,
    "SOFT_UPDATE": True, 
    "SOFT_UPDATE_TAU": 5e-3,
    "REWARD_SCALING_FACTOR": 1e-3,
    "PREFILL_BUFFER": False, # not to be used always
    "DEVICE": "cuda",
#    "STEPS_LIMIT": 100,
}

# the only question here is on the buffer size

# exp = UntilWinExperiment(params)
# result = exp.run()
# print("Method {} took an average of {:.2f} episodes".format(params["METHOD"], result))

# works ok

new_params = {
    "PRIORITIZED_REPLAY": True,
    "PRIO_REPLAY_ALPHA": 0.6,  
    "PRIO_REPLAY_BETA_START": 0.4,   
    "PRIO_REPLAY_BETA_FRAMES": 10000,
}
params.update(new_params)

results = []
methods = [True, False]
for method in methods:
    params["PRIORITIZED_REPLAY"] = method
    exp = UntilWinExperiment(params)
    result = exp.run()
    results.append(result)

for method, result in zip(methods, results):
    print("Method {} took an average of {:.2f} episodes".format(method, result))

# 32, 52, 100 (was getting close, maybe around 110)
