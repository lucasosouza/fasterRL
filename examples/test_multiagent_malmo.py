import sys
sys.path.append("../../")

from fasterRL.common.experiment import MultiAgentExperiment

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
    "MEAN_REWARD_BOUND": 950,
    "NUM_TRIALS": 1,
    "MAX_EPISODES": 3000,
    "EPSILON_DECAY_LAST_FRAME": 4000, # 4000
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
    "NUM_AGENTS": 2
}

# test sharing as well
params["SHARING"] = True
params["SHARE_BATCH_SIZE"] = 128

# test focused sharing as well


sharings = [True]
for sharing in sharings:
    params["SHARING"] = sharing
    exp = MultiAgentExperiment(params)
    result = exp.run()        
    print("Sharing: {}".format(sharing))
    for idx_a, res in enumerate(result):
        print("Method {} took an average of {:.2f} episodes for agent {}".format(
            params["METHOD"], res, idx_a))    


# multiagent ok
# multiagent with experience sharing ok
# multiagent with focused experienec sharing