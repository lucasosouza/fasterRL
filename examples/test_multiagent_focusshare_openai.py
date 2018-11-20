
import sys
sys.path.append("../../")

from fasterRL.common.experiment import MultiAgentExperiment

params = {
    "PLATFORM": "openai",
    "ENV_NAME": "CartPole-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "SimpleValueNetwork",
    "REPORTING_INTERVAL": 10,
    "LOG_LEVEL": 1, 
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 199,
    "NUM_TRIALS": 10,
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
    "FOCUSED_SHARING": True,
    "SHARE_BATCH_SIZE": 128,
    "FOCUSED_SHARING_THRESHOLD": 3
}

sharings = [True]
for sharing in sharings:
    params["FOCUSED_SHARING"] = sharing
    exp = MultiAgentExperiment(params)
    result = exp.run()        
    print("Sharing: {}".format(sharing))
    for idx_a, res in enumerate(result):
        print("Method {} took an average of {:.2f} episodes for agent {}".format(
            params["METHOD"], res, idx_a))

# Method DQN took an average of 263.90 episodes

# DQN without regular experience sharing took:
# Method DQN took an average of 263.27 episodes for agent 0
# Method DQN took an average of 254.87 episodes for agent 1

# DQN with regular experience sharing took:
# Method DQN took an average of 179.00 episodes for agent 0
# Method DQN took an average of 208.33 episodes for agent 1

# DQN with focused experience sharing took:
# Method DQN took an average of 178.00 episodes for agent 0
# Method DQN took an average of 164.10 episodes for agent 1

# can see an improvement here, but still need to test it in minecraft
# the last moving piece here is the sharing with prioritized experience replay




