from fasterRL.agents import *
from fasterRL.common.logger import *
from fasterRL.common.environment import *

import os
from datetime import datetime
from time import sleep, time
import json
from collections import namedtuple, defaultdict
import numpy as np


AgentExperiment = namedtuple('AgentExperiment', field_names=['env', 'agent', 'logger'])

class BaseExperiment:

    def __init__(self, params, experiment_name=None, experiment_group=None):

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
        if not experiment_name:
            experiment_id = "-".join([params["METHOD"], params["ENV_NAME"], now])
        else:
            experiment_id = experiment_name

        # dumps json with experiment hyperparameters
        if experiment_group:
            # if there is a group, save it in a folder with the groups name
            os.makedirs(os.path.join(log_root, "logs", experiment_group), exist_ok=True)
            params_log_path = os.path.join(log_root, "logs", experiment_group, experiment_id + ".json")
            self.log_dir = os.path.join(log_root, "runs", experiment_group, experiment_id)
        else:
            params_log_path = os.path.join(log_root, "logs", experiment_id + ".json")
            self.log_dir = os.path.join(log_root, "runs", experiment_id)

        with open(params_log_path, "w") as f:
            json.dump(params, f)

        # log paths, tensorboard agents specifics and trial overall json
        # don't use groups for runs yet until the impact on tensorboard is clear
        # self.log_dir = os.path.join(log_root, "runs", experiment_id)

        # set path for local results
        if experiment_group:
            os.makedirs(os.path.join(log_root, "results", experiment_group), exist_ok=True)
            local_log_path = os.path.join(log_root, "results", experiment_group, experiment_id + '.json')
        else:
            local_log_path = os.path.join(log_root, "results", experiment_id + '.json')

        self.num_trials = 1
        if "NUM_TRIALS" in self.params:
            self.num_trials = self.params["NUM_TRIALS"]

        self.num_episodes = 1
        if "NUM_EPISODES" in self.params:
            self.num_episodes = self.params["NUM_EPISODES"]

        self.steps_limit = None
        if "STEPS_LIMIT" in self.params:
            self.steps_limit = self.params["STEPS_LIMIT"]

        # also uses a log level, for things above the agent level
        self.log_level = 2
        if "LOG_LEVEL" in params:
            self.log_level = params["LOG_LEVEL"]            
        
        self.reporting_interval = 1
        if "REPORTING_INTERVAL" in params:
            self.reporting_interval = params["REPORTING_INTERVAL"]

        self.prefill_buffer = False
        if "PREFILL_BUFFER" in params:
            self.prefill_buffer = params["PREFILL_BUFFER"]

        # define methods for agent, env and logger
        self.agent_method = eval(params["METHOD"])
        self.env_method = BaseEnv
        self.logger_method = BaseLogger

        self.exp_logger = ExperimentLogger(local_log_path)

        if self.log_level > 1:
            print("Initializing experiment: ", experiment_id)

    def run(self):

        # training loop
        for trial in range(self.num_trials):
            t0 = time()
            episodes, avg_reward, avg_steps = self.run_trial(trial)

            # update logger
            time_spent = time() - t0
            self.exp_logger.update(time_spent, episodes, avg_reward, avg_steps)

        # print to screen
        if self.log_level > 1:
            self.exp_logger.report()
            self.exp_logger.save()
                       
        # ensuring backwards compatibility
        return np.mean(self.exp_logger.episodes_to_complete)

    def init_instances(self, trial, alias="agent", color=-1):

        # instantiate env, logger and agent for every trial
        env = self.env_method(self.params) # ok 
        agent = self.agent_method(self.params) # ok
        agent.set_environment(env)
        agent.set_alias(alias)
        logger = self.logger_method(self.params, self.log_dir, agent, trial, color) # ok

        if self.prefill_buffer:
            agent.fill_buffer()

        return AgentExperiment(env, agent, logger)

    def run_trial(self, trial):

        env, agent, logger = self.init_instances(trial)

        # training
        logger.start_training()
        for episode in range(self.num_episodes):
            self.run_episode(agent, logger)
        logger.end_training()

        return logger.episode_count, np.mean(logger.rewards), np.mean(logger.steps)

    def run_episode(self, agent, logger):

        logger.start_episode()
        agent.reset()
        episode_complete = False

        # user can set a step limit - can add this as a params
        if self.steps_limit:
            while not episode_complete and logger.steps_count < self.steps_limit:
                print(logger.steps_count, self.steps_limit)
                episode_complete = agent.play_step()
                # somehow it is logging 1 to 2 extra steps in Malmo
                # see if this is ocurring in other environemnts
                # what can be happening is agent issues the step, but the environment doesn't accept it
                logger.log_step()
        else:
            while not episode_complete:
                episode_complete = agent.play_step()
                logger.log_step()

        logger.log_episode()

class UntilWinExperiment(BaseExperiment):
    """ agent plays until it wins. may define a max number of episodes """ 

    def __init__(self, params, experiment_name=None, experiment_group=None):
        super(UntilWinExperiment, self).__init__(params, experiment_name, experiment_group)

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

        # can print results here, besides from returning
        return logger.episode_count, np.mean(logger.rewards), np.mean(logger.steps)


class MultiAgentExperiment(UntilWinExperiment):
    """ Two or more agents plays sequentially 
        Modifications are done only to run and run trial functions
    """

    def __init__(self, params, experiment_name=None, experiment_group=None):
        super(MultiAgentExperiment, self).__init__(params, experiment_name, experiment_group)

        # unpack params
        self.num_agents = 1
        if "NUM_AGENTS" in self.params:
            self.num_agents = self.params["NUM_AGENTS"]

        self.sharing = False
        if "SHARING" in self.params:
            self.sharing = self.params["SHARING"]

        self.focused_sharing = False
        if "FOCUSED_SHARING" in self.params:
            self.focused_sharing = self.params["FOCUSED_SHARING"]

        if self.sharing or self.focused_sharing:
            self.share_batch_size = 128
            if "SHARE_BATCH_SIZE" in self.params:
                self.share_batch_size = self.params["SHARE_BATCH_SIZE"]

        if self.focused_sharing:
            self.sharing = False # turn of regular sharing, one or the other
            self.focused_sharing_threshold = 3
            if "FOCUSED_SHARING_THRESHOLD" in self.params:
                self.focused_sharing_threshold = self.params["FOCUSED_SHARING_THRESHOLD"]

    def run(self):
        """ Modified to return the average number of episodes to finish 
            If not finished, return max (an oversimplification)

            Adaptations for multiagent.
        """

        t0 = time()
        for trial in range(self.num_trials):
            multiagent_num_episodes = self.run_trial(trial)

            # update logger
            time_spent = (time() - t0) / len(multiagent_num_episodes)
            for episodes, avg_reward, avg_steps, exp_received in multiagent_num_episodes:
                self.exp_logger.update(time_spent, episodes, avg_reward, avg_steps, exp_received)

        # print to screen and save logger results
        if self.log_level > 1:
            self.exp_logger.report()
            self.exp_logger.save()

        # ensuring backwards compatibility
        return np.mean(self.exp_logger.episodes_to_complete)


    def run_trial(self, trial):
        """ Modified to run until problem is solved or number of max episodes is reached

            Adaptations to multiagent. 
            (loops are fast while agents number is low, hence are repeated to make the code more readable)
        """

        agents = []

        # initialize all agents
        for idx_a in range(self.num_agents):
            agents.append(self.init_instances(trial, alias="agent"+str(idx_a), color=idx_a))

        # start training
        for a in agents:
            a.logger.start_training()

        # initialize a variable to count the number of experiences shared
        self.experiences_shared = 0

        # alternate between agents to run episodes
        while sum([a.agent.completed for a in agents]) != len(agents):
            # one round of training
            for a in agents:
                if not a.logger.is_solved() and a.logger.episode_count < self.max_episodes:
                    self.run_episode(a.agent, a.logger)
                else:
                    a.agent.completed = True
            # one round of experience sharing
            if self.sharing:
                self.share(agents) # replace by full since need logger as well
            elif self.focused_sharing:
                self.focus_share(agents)

        # end training
        for a in agents:
            a.logger.end_training()

        return [(a.logger.episode_count, np.mean(a.logger.rewards), np.mean(a.logger.steps), a.logger.experiences_received) for a in agents]

    def share(self, agents):
        """ Allow transfer between N agents
            Ideally should not talk directly to buffer (currently does)

        """

        # init list to store all transfer batches
        transfer_batches = []

        # select experiences to share
        for a in agents:
            transfer_batch = a.agent.buffer.select_batch(self.share_batch_size)
            transfer_batches.append(transfer_batch)

        # receive experiences from all other agents
        for idx_a, a in enumerate(agents):
            # agents who have completed episodes do not need more experiences
            if not a.agent.completed:
                batch_indices = list(range(len(agents)))
                batch_indices.pop(idx_a) # agent should not receive his own experiences
                for idx_b in batch_indices:
                    a.agent.buffer.receive(transfer_batches[idx_b])
                    a.logger.experiences_received += len(transfer_batches[idx_b])

        if self.log_level > 4:
            print("Number of experiences transferred: {}".format([len(tb) for tb in transfer_batches]))

    def focus_share(self, agents):
        """ For now accomodates two agents. Increase functionalities later 
            can make this for N number of agents, but need to define rules.
            Ideally should not talk directly to buffer

            Difference from share: student has a mask
            This mask is used to let the teacher nows which experiences matter the most
        """

        # each agent puts forward a request wit experiences wanted
        transfer_requests = [] 
        for a in agents:
            transfer_request = a.agent.buffer.identify_unexplored(threshold=self.focused_sharing_threshold)
            transfer_requests.append(transfer_request)

        # for each request, gather transfers
        transfer_batches = defaultdict(list)
        for idx_r, request in enumerate(transfer_requests):
            # all agents respond to request
            for idx_a, a in enumerate(agents):
                # unless its the agents own request
                if idx_r != idx_a:
                    transfer_batch = a.agent.buffer.select_batch_with_mask(self.share_batch_size, request)
                    transfer_batches[idx_r].append(transfer_batch)

        # each agent receives the transfers sent to it
        tb_sizes = []
        for idx_a, batches in transfer_batches.items():
            a = agents[idx_a]
            num_experiences_received = 0
            # agents who have completed episodes do not need more experiences
            if not a.agent.completed:
                for batch in batches:              
                    a.agent.buffer.receive(batch)
                    a.logger.experiences_received += len(batch)
                    num_experiences_received += len(batch)
            tb_sizes.append(num_experiences_received)

        # removed tb sizes - replaced by logger

        # report
        if self.log_level > 4:
            print("Number of experiences transferred: {}".format(tb_sizes))


"""

# can stop from receiving the experience in the first round
# but not in the second
# how do I stop an agent from receiving identical experiences?
# I can maybe come up with a hash kind of algorithm to identify an experience
# and how do I check all hashes before I add an experience?
# this is something to look at later

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

