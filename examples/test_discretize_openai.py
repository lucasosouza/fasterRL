import sys
sys.path.append("../../")

from fasterRL.common.experiment import UntilWinExperiment

# "ENV_NAME": "CartPole-v0",
params = {
    "PLATFORM": "openai",
    "ENV_NAME": "CartPole-v0",
    "METHOD": "QLearning",
    "REPORTING_INTERVAL": 100,
    "LOG_LEVEL": 2,
    "LOGGER_METHOD": "TDLogger",
    "NUMBER_EPISODES_MEAN": 10,
    "MEAN_REWARD_BOUND": 195,
    "NUM_TRIALS": 3, # 10
    "MAX_EPISODES": 10000,
    "EPSILON_DECAY_LAST_FRAME": 50000, # is thi sok
    "EPSILON_START": 1.0,
    "EPSILON_FINAL": 0.02, # 0.02
    "LEARNING_RATE": 0.05,
    "GAMMA": 0.99,
    "DISCRETIZE_STATE": True,
    "DISCRETIZE_BIN_SIZE": 10,
    "DISCRETIZE_STATE_TYPE": 'unitary',
    "DISCRETIZE_STATE_SAMPLING_SIZE": 10000,
}

exp = UntilWinExperiment(params)
exp.run()

# OVERALL 

# with more exploration it starts to learn
# would need a non linear decay on epsilon to be able to stay a long time at a low epsilon rate

# problem being solved with a lower learning rate, and higher exploration, in between 1600 and 3400 episodes and 2 seconds (compared to 150 episodes and 10 seconds from DQN best results)

# with TRUE SAMPLING

# bin size too little (<=5) and too much (>=30), agent does not learn
# best results with 10

# how much sampling is done to define the bins also affect performance. <=100 less is too little, about 1000 already seem optimal. 

# COMPARISON with discretization

# with UNITARY discretization: 3174, 4341, 5226, None, 1541
# with TRUE SAMPLING discreti: None, 9072, None, None, None

# best results from UNITARY discretization, even with infinite ranges in the state space