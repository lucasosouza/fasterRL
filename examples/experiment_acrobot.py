


import gym
gym.logger.set_level(40)

from fasterrl.common.experiment import *

params = {
    "PLATFORM": "openai",
    "ENV_NAME": "CartPole-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "SimpleValueNetwork",
    "REPORTING_INTERVAL": 10,
    "RENDER": False,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 199,
    "NUM_TRIALS": 10,
    "MAX_EPISODES": 3000,
    "EPSILON_DECAY_LAST_FRAME": 3000, # 4000
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0, # 0.02
    "LEARNING_RATE": 2e-4, # 1e-3
    "GAMMA": 0.99, # 0.99
    "REPLAY_BATCH_SIZE": 128, # 32
    "EXPERIENCE_BUFFER_SIZE": 10000, # 5000
    "GRADIENT_CLIPPING": False,
    "DOUBLE_QLEARNING": True,
    "SOFT_UPDATE": True,
    "SOFT_UPDATE_TAU": 5e-3,
    "REWARD_SCALING_FACTOR": 1e-3,
    "PREFILL_BUFFER": False,
    "DEVICE": "cuda",
}

# add prioritized replay
new_params = {
    "PRIORITIZED_REPLAY": True,
    "PRIO_REPLAY_ALPHA": 0.6,
    "PRIO_REPLAY_BETA_START": 0.4,
    "PRIO_REPLAY_BETA_FRAMES": 50000,
}
params.update(new_params)

# test single agent, with and without replay
options = [
    {"PRIORITIZED_REPLAY": False},
    {"PRIORITIZED_REPLAY": True},
]

results = []
for option in options:
    params.update(option)
    results.append(UntilWinExperiment(params).run())

for result, option in zip(results, options):
    print("Average of {:.2f} episodes for option {}".format(result, option))

# test single agent



# then test multiagent


