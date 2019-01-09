import sys
sys.path.append("../../")

from fasterRL.common.experiment import UntilWinExperiment, MultiAgentExperiment

exp_group = __file__[:-3]
NUM_SAMPLES = 30

# base DQN
dqn = {
    "PLATFORM": "marlo",
    "ENV_NAME": "MarLo-FindTheGoal-v0",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "DeepQNetwork",
    "REPORTING_INTERVAL": 1,
    "RENDER": True,
    "LOG_LEVEL": 2, #debugging level
    "NUMBER_EPISODES_MEAN": 5,
    "MEAN_REWARD_BOUND": 0.40,
    "NUM_TRIALS": NUM_SAMPLES,
    "MAX_EPISODES": 100,
    "EPSILON_DECAY_LAST_FRAME": 600, # 4000
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0.01, # 0.02
    "LEARNING_RATE": 2e-4, # 1e-4 # not sure if change?
    "GAMMA": 0.97, # 0.99
    "REPLAY_BATCH_SIZE": 128, # 32
    "EXPERIENCE_BUFFER_SIZE": 10000, # 5000
    "GRADIENT_CLIPPING": False,
    "DOUBLE_QLEARNING": True,
    "SOFT_UPDATE": True, 
    "SOFT_UPDATE_TAU": 5e-3,
    "PREFILL_BUFFER": False, # not to be used always
    "DEVICE": "cuda",
}



prio = {
    "PRIORITIZED_REPLAY": True,
    "PRIO_REPLAY_ALPHA": 0.6,  
    "PRIO_REPLAY_BETA_START": 0.4,   
    "PRIO_REPLAY_BETA_FRAMES": 10000,
}

dqn_prio = dqn.copy()
dqn_prio.update(prio)

sharing = {
    "NUM_AGENTS": 2,
    "SHARE_BATCH_SIZE": 128,
    "SHARING": True,
    "NUM_TRIALS": int(NUM_SAMPLES/2),  
}

dqn_sharing = dqn.copy()
dqn_sharing.update(sharing)

focus = {
    "FOCUSED_SHARING": True,
    "FOCUSED_SHARING_THRESHOLD": 5,    
}

dqn_focus_sharing = dqn_sharing.copy()
dqn_focus_sharing.update(focus)


# others
dqn_prio_sharing = dqn_sharing.copy()
dqn_prio_sharing.update(prio)

dqn_prio_focus_sharing = dqn_sharing.copy()
dqn_prio_focus_sharing.update(prio)
dqn_prio_focus_sharing.update(focus)

## prepare the experiment
exp_group = exp_group
experiments = {
    'dqn': dqn,
    'dqn_prio': dqn_prio,
	'dqn_sharing': dqn_sharing,
	'dqn_prio_sharing': dqn_prio_sharing,
	'dqn_focus_sharing': dqn_focus_sharing,
    'dqn_prio_focus_sharing': dqn_prio_focus_sharing,
}

for exp_name, params in reversed(list(experiments.items())):
    print(exp_name, params)
    if 'sharing' in exp_name:
        exp = MultiAgentExperiment(params, exp_name, exp_group)
    else:
        exp = UntilWinExperiment(params, exp_name, exp_group)
    exp.run()
