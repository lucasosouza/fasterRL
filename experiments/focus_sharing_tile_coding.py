"""
Experiment:
test using tile coding
code does not run out of the box
might not be too smart to waste time fixing this now

"""

import sys
sys.path.append("../../")
from fasterRL.common.experiment import UntilWinExperiment, MultiAgentExperiment
from math import ceil
import numpy as np

exp_group = __file__[:-3]
NUM_SAMPLES = 100

# base DQN
dqn = {
    # general params
    "PLATFORM": "openai",
    "METHOD": "DQN",
    "LOGGER_METHOD": "DQNLogger",
    "NETWORK_TYPE": "SimpleValueNetwork",
    "REPORTING_INTERVAL": 40,
    "LOG_LEVEL": 2,
    "NUM_TRIALS": NUM_SAMPLES,

    # specific environment parameters
    "ENV_NAME": "CartPole-v0",
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 199,
    "MAX_EPISODES": 1000,

    # algorithm parameters
    "EPSILON_DECAY_LAST_FRAME": 4000,
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0,
    "LEARNING_RATE": 1e-3,
    "GAMMA": 0.99,
    "REPLAY_BATCH_SIZE": 32,
    "EXPERIENCE_BUFFER_SIZE": 20000,
    "GRADIENT_CLIPPING": False,
    "DOUBLE_QLEARNING": True,
    "SOFT_UPDATE": True, 
    "SOFT_UPDATE_TAU": 5e-3,
}

# prio = {
#     "PRIORITIZED_REPLAY": True,
#     "PRIO_REPLAY_ALPHA": 0.6,  
#     "PRIO_REPLAY_BETA_START": 0.4,   
#     "PRIO_REPLAY_BETA_FRAMES": 10000,
# }

# dqn_prio = dqn.copy()
# dqn_prio.update(prio)

sharing = {
    "NUM_AGENTS": 2,
    "SHARE_BATCH_SIZE": 128,
    "SHARING": True,
    "NUM_TRIALS": ceil(NUM_SAMPLES/2),  
}

dqn_sharing = dqn.copy()
dqn_sharing.update(sharing)

focus = {
    "FOCUSED_SHARING": True,
    "FOCUSED_SHARING_THRESHOLD": 10,    
	"DISCRETIZE_STATE": True,
	"DISCRETIZE_BIN_SIZE": 10,
	"DISCRETIZE_TYPE": 'unitary',
	"WITH_TILES": True    
}

dqn_focus_sharing = dqn_sharing.copy()
dqn_focus_sharing.update(focus)

# # others
# dqn_prio_sharing = dqn_sharing.copy()
# dqn_prio_sharing.update(prio)

# dqn_prio_focus_sharing = dqn_sharing.copy()
# dqn_prio_focus_sharing.update(prio)
# dqn_prio_focus_sharing.update(focus)

## prepare the experiment
# experiments = {
#     'dqn': dqn,
#     'dqn_prio': dqn_prio,
#     'dqn_sharing': dqn_sharing,
#     'dqn_prio_sharing': dqn_prio_sharing,
#     'dqn_focus_sharing': dqn_focus_sharing,
#     'dqn_prio_focus_sharing': dqn_prio_focus_sharing,
# }

tiles = {
	1: [0],
	5: list(np.arange(-0.10, 0.11, 0.05)),
	10: list(np.arange(-0.10, 0.11, 0.02)),
	20: list(np.arange(-0.10, 0.11, 0.01)),
	40: list(np.arange(-0.20, 0.21, 0.01))
}



# 1 - is just insurance

for tiles, list_tiles in tiles.items():

    exp_name = 'dqn_focus_sharing_tiles' + str(tiles)
    exp_params = dqn_focus_sharing.copy()
    exp_params["TILE_OFFSETS"] = list_tiles

    print(exp_name, exp_params)
    exp = MultiAgentExperiment(exp_params, exp_name, exp_group)
    exp.run()