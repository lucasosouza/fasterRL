import sys
sys.path.append("../../")

# remove warnings
#import warnings
#warnings.filterwarnings("ignore")
import gym
gym.logger.set_level(40)


from fasterRL.common.experiment import MultiAgentExperiment

params = {
    "PLATFORM": "openai",
    "ENV_NAME": "CartPole-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "SimpleValueNetwork",
    "REPORTING_INTERVAL": 10,
    "RENDER": False,
    "LOG_LEVEL": 1,
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 199,
    "NUM_TRIALS": 30,
    "MAX_EPISODES": 300,
    "EPSILON_DECAY_LAST_FRAME": 3000, # 4000
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0, # 0.02
    "LEARNING_RATE": 2e-4, # 1e-3
    "GAMMA": 0.99, # 0.99
    "REPLAY_BATCH_SIZE": 128, # 32
    "EXPERIENCE_BUFFER_SIZE": 50000, # 5000
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

# add sharing
new_params = {
    "NUM_AGENTS": 2,
    "SHARING": True,
    "SHARE_BATCH_SIZE": 128,
    "FOCUSED_SHARING": True,
    "FOCUSED_SHARING_THRESHOLD": 3,
}
params.update(new_params)

# test for different options
options = [
    {"SHARING": False, "PRIORITIZED_REPLAY": False},
    {"SHARING": False, "PRIORITIZED_REPLAY": True},
    {"SHARING": True, "PRIORITIZED_REPLAY": False},
    {"SHARING": True, "PRIORITIZED_REPLAY": True},
    {"FOCUSED_SHARING": True, "PRIORITIZED_REPLAY": False},
    {"FOCUSED_SHARING": True, "PRIORITIZED_REPLAY": True}
]

for option in options:
    params.update(option)
    exp = MultiAgentExperiment(params)
    res = exp.run()        
    print("Average episodes {:.2f}, {:.2f} for option {}".format(
        res[0], res[1], option))    

# multiagent ok
# multiagent with experience sharing ok

# will do a full round of text

# multiagent focused experience sharing ok - still issues I have to investigate relating to agent A sharing the same experience with agent B that it received in last round from agent B (so for B would only be duplicated)




