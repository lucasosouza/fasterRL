import sys
sys.path.append("../../")

from fasterRL.common.experiment import MultiAgentExperiment

params = {
    "PLATFORM": "openai-atari",
    "ENV_NAME": "Pong-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "DeepQNetwork",
    "REPORTING_INTERVAL": 1,
    "LOG_LEVEL": 2,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 0,
    "NUM_TRIALS": 1,
    "MAX_EPISODES": 1000,
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0,
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.99,
    "NUM_AGENTS": 2,
    "REPLAY_BATCH_SIZE": 32,
    "EXPERIENCE_BUFFER_SIZE": 5000,
    "GRADIENT_CLIPPING": False,
    "DOUBLE_QLEARNING": True,
    "SOFT_UPDATE": True, 
    "SOFT_UPDATE_TAU": 5e-3 
}

# test for focused sharing
params["FOCUSED_SHARING"] = True
params["SHARE_BATCH_SIZE"] = 32
params["FOCUSED_SHARING_THRESHOLD"] = 3
params["LOGGER_METHOD"] = "StepLogger"
params["STEPS_LIMIT"] = 36 # set low steps limit to induce sharing faster for testing

exp = MultiAgentExperiment(params)
result = exp.run()
for idx_a, res in enumerate(result):
    print("Method {} took an average of {:.2f} episodes for agent {}".format(
        params["METHOD"], res, idx_a))

# expectations: print one episode complete for both agents. if taking 3 minutes each, should take around 5 minutes

# without focused sharing
# Episode 1 Mean Reward: -21.000 Mean Steps: 1226.000 Speed: 192.058 sec/ep Frame: 1226
# Episode 1 Mean Reward: -20.000 Mean Steps: 1276.000 Speed: 201.119 sec/ep Frame: 1276


# with focused sharing
# the true test is when they try to share experience
# will be a bummer if this breaks
# because one episode is going to take forever


