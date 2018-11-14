from agents import *

methods = {
    "DQN": DQN.DQN_experiment,
    "DQN_Malmo": DQN_Malmo.DQN_experiment,
    "DQN_Malmo_Para": DQN_Malmo_Para.DQN_experiment,
    "DQN_Malmo_Seq": DQN_Malmo_Seq.DQN_experiment,
    "Simple_Malmo": Simple_Malmo.run_experiment,
    "DQN_Marlo": DQN_Marlo.DQN_experiment
}