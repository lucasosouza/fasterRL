from fasterRL.agents import *
from fasterRL.common.logger import *
from fasterRL.common.environment import *

import os
from datetime import datetime
from time import sleep
import json
from collections import namedtuple


AgentExperiment = namedtuple('AgentExperiment', field_names=['env', 'agent', 'logger'])

class BaseExperiment:

    def __init__(self, params):

        self.params = params

        ### Set experiment path

        # identify log directory
        if "LOG_DIR" in os.environ:
            log_root = os.environ["LOG_DIR"]
        else:
            log_root = "./"

        # create an ID for the experiment
        now = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S-%f")[:-4]
        sleep(0.01) # intentional delay, avoids having file with exact name
        experiment_id = "-".join([params["METHOD"], params["ENV_NAME"], now])

        # dumps json with experiment hyperparameters
        with open(os.path.join(log_root, "logs", experiment_id + ".json"), "w") as f:
            json.dump(params, f)
        print("Initializing experiment: ", experiment_id)

        self.log_dir = os.path.join(log_root, "runs", experiment_id)
        # save path for local method
        self.local_log_path = os.path.join(log_root, "results", experiment_id + '.json')

        # unpack params
        self.num_trials = 1
        if "NUM_TRIALS" in self.params:
            self.num_trials = self.params["NUM_TRIALS"]

        self.num_episodes = 1
        if "NUM_EPISODES" in self.params:
            self.num_episodes = self.params["NUM_EPISODES"]

        # define methods for agent, env and logger
        self.agent_method = eval(params["METHOD"])
        self.env_method = BaseEnv
        self.logger_method = BaseLogger

    def run(self):

        # training loop
        for trial in range(self.num_trials):
            self.run_trial(trial)

    def init_instances(self, trial, alias="agent"):

        # instantiate env, logger and agent for every trial
        env = self.env_method(self.params) # ok 
        agent = self.agent_method(self.params) # ok
        agent.set_environment(env)
        agent.set_alias(alias)
        logger = self.logger_method(self.params, self.log_dir, agent, trial) # ok

        return AgentExperiment(env, agent, logger)

    def run_trial(self, trial):

        env, agent, logger = self.init_instances(trial)

        # training
        logger.start_training()
        for episode in range(self.num_episodes):
            self.run_episode(agent, logger)
        logger.end_training()

    def run_episode(self, agent, logger):

        logger.start_episode()
        agent.reset()
        episode_complete = False
        while not episode_complete:
            episode_complete = agent.play_step()
            logger.log_step()
        logger.log_episode()

class UntilWinExperiment(BaseExperiment):
    """ agent plays until it wins. may define a max number of episodes """ 

    def __init__(self, params):
        super(UntilWinExperiment, self).__init__(params)

        # unpack params
        self.max_episodes = 1
        if "MAX_EPISODES" in self.params:
            self.max_episodes = self.params["MAX_EPISODES"]

        self.logger_method = WinLogger
        if "LOGGER_METHOD" in params:
            self.logger_method = eval(params["LOGGER_METHOD"])

    # override other methods
    def run_trial(self, trial):
        """ Modified to run until problem is solved or number of max episodes is reached
        """

        env, agent, logger = self.init_instances(trial)

        logger.start_training()
        while not logger.is_solved() and logger.episode_count < self.max_episodes:
            self.run_episode(agent, logger)
        logger.end_training()

        return logger.episode_count

    def run(self):
        """ Modified to return the average number of episodes to finish 
            If not finished, return max (an oversimplification)
        """

        all_trial_episodes = []
        for trial in range(self.num_trials):
            num_episodes = self.run_trial(trial)
            all_trial_episodes.append(num_episodes)

        return sum(all_trial_episodes)/len(all_trial_episodes)



class MultiAgentExperiment(UntilWinExperiment):
    """ Two or more agents plays sequentially 
        Modifications are done only to run and run trial functions
    """


    def __init__(self, params):
        super(MultiAgentExperiment, self).__init__(params)

        # unpack params
        self.num_agents = 1
        if "NUM_AGENTS" in self.params:
            self.num_agents = self.params["NUM_AGENTS"]

    def run(self):
        """ Modified to return the average number of episodes to finish 
            If not finished, return max (an oversimplification)

            Adaptations for multiagent.
        """

        all_trial_episodes = []
        for idx_a in range(self.num_agents):
            all_trial_episodes.append([])

        for trial in range(self.num_trials):
            multiagent_num_episodes = self.run_trial(trial)
            for idx_a, num_episodes in enumerate(multiagent_num_episodes):
                all_trial_episodes[idx_a].append(num_episodes)

        return [sum(l)/len(l) for l in all_trial_episodes]  

    def run_trial(self, trial):
        """ Modified to run until problem is solved or number of max episodes is reached

            Adaptations to multiagent. 
            (loops are fast while agents number is low, hence are repeated to make the code more readable)
        """

        agents = []

        # initialize all agents
        for idx_a in range(self.num_agents):
            agents.append(self.init_instances(trial, alias="agent"+str(idx_a)))

        # start training
        for a in agents:
            a.logger.start_training()

        # alternate between agents to run episodes
        solved = {}
        while sum(solved.values()) != len(agents):
            for a in agents:
                if not a.logger.is_solved() and a.logger.episode_count < self.max_episodes:
                    self.run_episode(a.agent, a.logger)
                else:
                    solved[a.agent.alias] = True

        # end training
        for a in agents:
            a.logger.end_training()
    
        return [a.logger.episode_count for a in agents]













"""

to consider it later:

# this is not thought of to run in parallel
# how would it run in parallel
# should I prepare the library for it?
# yes, I should, that is the only answer I can think of as of now 
# however, I still think there should be two separate logs
# if that is so, then the experiment id generation goes back up

# can also have multiple experiments class
# which would allow for multiple ways of running this


maybe specify when problem is not solved instead of reporting max episodes

"""

