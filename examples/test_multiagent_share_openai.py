



from fasterrl.common.experiment import MultiAgentExperiment

params = {
    "PLATFORM": "openai",
    "ENV_NAME": "CartPole-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "SimpleValueNetwork",
    "REPORTING_INTERVAL": 10,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 130,# 199
    "NUM_TRIALS": 2,
    "MAX_EPISODES": 3000,
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0,
    "LEARNING_RATE": 1e-3,
    "GAMMA": 0.99,
    "REPLAY_BATCH_SIZE": 32,
    "EXPERIENCE_BUFFER_SIZE": 5000,
    "GRADIENT_CLIPPING": False,
    "DOUBLE_QLEARNING": True,
    "SOFT_UPDATE": True,
    "SOFT_UPDATE_TAU": 5e-3,
    "NUM_AGENTS": 2,
    "SHARING": True,
    "SHARE_BATCH_SIZE": 100,
}

sharings = [True]
for sharing in sharings:
    params["SHARING"] = sharing
    exp = MultiAgentExperiment(params)
    result = exp.run()
    print("Sharing: {}".format(sharing))
    print("Method {} took an average of {:.2f} episodes".format(params["METHOD"], result))

# Method DQN took an average of 263.90 episodes

# DQN with regular experience sharing took:
# Method DQN took an average of 179.00 episodes for agent 0
# Method DQN took an average of 208.33 episodes for agent 1

# that is significantly faster
# test it in minecraft to see if improvement holds

# DQN without regular experience sharing took:
# Method DQN took an average of 263.27 episodes for agent 0
# Method DQN took an average of 254.87 episodes for agent 1




