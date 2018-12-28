import sys
sys.path.append("../../")

from fasterRL.common.experiment import UntilWinExperiment, MultiAgentExperiment

params = {
    "PLATFORM": "marlo",
    "ENV_NAME": "MarLo-FindTheGoal-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "DeepQNetwork",
    "REPORTING_INTERVAL": 1,
    "RENDER": True,
    "LOG_LEVEL": 2, #debugging level
    "NUMBER_EPISODES_MEAN": 5,
    "MEAN_REWARD_BOUND": 0.35,
    "NUM_TRIALS": 3,
    "MAX_EPISODES": 150,
    "EPSILON_DECAY_LAST_FRAME": 600, # 4000
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
    "PREFILL_BUFFER": False, # not to be used always
    "DEVICE": "cuda",
    "PRIORITIZED_REPLAY": True,
    "PRIO_REPLAY_ALPHA": 0.6,  
    "PRIO_REPLAY_BETA_START": 0.4,   
    "PRIO_REPLAY_BETA_FRAMES": 10000,
}

new_params = {
    "NUM_AGENTS": 2,
    "SHARING": True,
    "SHARE_BATCH_SIZE": 128,
    "FOCUSED_SHARING": True,
    "FOCUSED_SHARING_THRESHOLD": 3,
}
params.update(new_params)

# exp = UntilWinExperiment(params)
exp = MultiAgentExperiment(params)
result = exp.run()

# test after the prio replay version


"""
Episode 1 | Avg Reward: -1.44 | Running Mean: -1.44 | Avg Steps: 169.00 | Ep.Speed: 28.87 sec/ep | Steps p/s 5.85 | Total steps: 169
Episode 1 | Avg Reward: -1.03 | Running Mean: -1.03 | Avg Steps: 134.00 | Ep.Speed: 12.34 sec/ep | Steps p/s 10.86 | Total steps: 134
Number of experiences transferred: [128, 128]
Episode 2 | Avg Reward: 0.36 | Running Mean: -0.54 | Avg Steps: 12.00 | Ep.Speed: 17.61 sec/ep | Steps p/s 0.68 | Total steps: 181
Episode 2 | Avg Reward: 0.41 | Running Mean: -0.31 | Avg Steps: 9.00 | Ep.Speed: 15.05 sec/ep | Steps p/s 0.60 | Total steps: 143
Number of experiences transferred: [128, 128]
Episode 3 | Avg Reward: 0.44 | Running Mean: -0.21 | Avg Steps: 5.00 | Ep.Speed: 11.49 sec/ep | Steps p/s 0.44 | Total steps: 186
Episode 3 | Avg Reward: 0.48 | Running Mean: -0.05 | Avg Steps: 4.00 | Ep.Speed: 14.27 sec/ep | Steps p/s 0.28 | Total steps: 147
Number of experiences transferred: [128, 128]
Episode 4 | Avg Reward: 0.45 | Running Mean: -0.05 | Avg Steps: 6.00 | Ep.Speed: 11.96 sec/ep | Steps p/s 0.50 | Total steps: 192
Episode 4 | Avg Reward: 0.39 | Running Mean: 0.06 | Avg Steps: 9.00 | Ep.Speed: 17.47 sec/ep | Steps p/s 0.52 | Total steps: 156
Number of experiences transferred: [128, 128]
Episode 5 | Avg Reward: 0.46 | Running Mean: 0.05 | Avg Steps: 4.00 | Ep.Speed: 12.19 sec/ep | Steps p/s 0.33 | Total steps: 196
Episode 5 | Avg Reward: 0.44 | Running Mean: 0.14 | Avg Steps: 6.00 | Ep.Speed: 12.83 sec/ep | Steps p/s 0.47 | Total steps: 162
Number of experiences transferred: [128, 128]
Episode 6 | Avg Reward: 0.49 | Running Mean: 0.44 | Avg Steps: 2.00 | Ep.Speed: 12.01 sec/ep | Steps p/s 0.17 | Total steps: 198
Episode 6 | Avg Reward: 0.43 | Running Mean: 0.43 | Avg Steps: 7.00 | Ep.Speed: 15.50 sec/ep | Steps p/s 0.45 | Total steps: 169
Number of experiences transferred: [128, 128]
Problem solved in 6 episodes
Problem solved in 6 episodes
"""