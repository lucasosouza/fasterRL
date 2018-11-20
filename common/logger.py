from time import time # should move import from here?
from tensorboardX import SummaryWriter
import os
import numpy as np
from termcolor import colored

class BaseLogger():

    def __init__(self, params, log_dir, agent, trial="", color=-1):
        """
            LOG LEVELS:
            1 - report nothing, just run
            2 - print to screen
            3 - log episode-wise variables
            4 - log step-wise variable
            5 - log specifics relevant for debugging
        """


        colors = ['blue','red','green','cyan','yellow','magenta','grey','white']
        self.color = colors[color]

        self.log_level = 2
        if "LOG_LEVEL" in params:
            self.log_level = params["LOG_LEVEL"]

        self.reporting_interval = 1
        if "REPORTING_INTERVAL" in params:
            self.reporting_interval = params["REPORTING_INTERVAL"]

        # only need the agent's variables, no need to keep ref to remaining
        self.agent = agent

        # initialize writers
        trial_dir = "".join([agent.alias, "-trial", str(trial)])            
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, trial_dir))

    def cprint(self, content):
        print(colored(content, self.color))

    def start_training(self):
        # controls number of episodes and frames. control time

        self.episode_count = 0
        self.frame_count = 0
        self.frame_start = time()

        # bookkeeping
        self.episode_reward = 0
        self.rewards = []
        self.steps = []


    def end_training(self):

        self.writer.close()

    def start_episode(self):
        # control number of steps. control time
        self.steps_count = 0
        self.episode_start = time()
        self.episode_reward = 0

    def log_step(self):

        # verify frame speed - should be similar to episode speed

        frame_speed = 1000 / (time() - self.frame_start)
        self.frame_count += 1
        self.steps_count += 1
        self.episode_reward += self.agent.step_reward

        if self.log_level > 3 :

            self.writer.add_scalar("speed/frame", frame_speed, self.frame_count)
            self.writer.add_scalar("reward/step", self.agent.step_reward, self.frame_count)

            # temporary workaround. may add several variables like this to logger instead of subclassing
            if hasattr(self.agent, "epsilon"):
                self.writer.add_scalar("epsilon", self.agent.epsilon, self.frame_count)

            self.frame_start = time()

    def log_episode(self):

        episode_speed = time() - self.episode_start
        self.episode_count += 1
        self.rewards.append(self.episode_reward)
        self.steps.append(self.steps_count)

        if self.log_level > 1 :

            if self.episode_count % self.reporting_interval == 0:
                self.cprint("Episode {} Mean Reward: {:.3f} Mean Steps: {:.3f} Speed: {:.3f} sec/ep Frame: {}".format(
                    self.episode_count, 
                    np.mean(self.rewards[-self.reporting_interval:]), 
                    np.mean(self.steps[-self.reporting_interval:]), 
                    episode_speed,
                    self.frame_count))

        if self.log_level > 2 :

            self.writer.add_scalar("reward/episode", self.episode_reward, self.episode_count)
            self.writer.add_scalar("steps", self.steps_count, self.episode_count)
            self.writer.add_scalar("speed/episode", episode_speed, self.episode_count)

            self.episode_start = time()

class WinLogger(BaseLogger):

    def __init__(self, params, log_dir, agent, trial="", color=-1):
        super(WinLogger, self).__init__(params, log_dir, agent, trial, color)

        # variables for until win experiment
        if "MEAN_REWARD_BOUND" in params:
            self.mean_reward_bound = params["MEAN_REWARD_BOUND"]
        else:
            print("Until win requires a MEAN_REWARD_BOUND to be defined")            

        if "NUMBER_EPISODES_MEAN" in params:
            self.number_episodes_mean = params["NUMBER_EPISODES_MEAN"]
        else:
            print("Until win requires a NUMBER_EPISODES_MEAN to be defined")

        # extra flag to avoid double reporting in multiagent. fix it later
        self.completed = False

    def is_solved(self):

        if self.episode_count >= self.number_episodes_mean:
            if np.mean(self.rewards[-self.number_episodes_mean:]) >= self.mean_reward_bound:
                if self.log_level > 1 and not self.completed:
                    self.cprint("Problem solved in {} episodes".format(self.episode_count))
                    self.completed = True
                return True
        return False

class DQNLogger(WinLogger):

    def log_step(self):
        super(DQNLogger, self).log_step()

        # level 5 - debugging
        if self.log_level > 4 :

            # maybe predict for when using GPU calculate differently
            q_vals = self.agent.calculate_q_vals().detach().numpy()[0]
            self.writer.add_scalar("q_value/min", min(q_vals), self.frame_count)
            self.writer.add_scalar("q_value/max", max(q_vals), self.frame_count)

class CrossEntropyLogger(WinLogger):

    def log_episode(self):
        super(CrossEntropyLogger, self).log_episode()

        # level 5 - debugging
        if self.log_level > 4 :

            self.writer.add_scalar("batch_reward/mean", self.agent.buffer.reward_mean, self.episode_count)
            self.writer.add_scalar("batch_reward/bound", self.agent.buffer.reward_bound, self.episode_count)

            self.writer.add_scalar("buffer_size", len(self.agent.buffer), self.episode_count)

            # report action probabilities to follow-up how policy evolves
            if len(self.agent.buffer) > 0:
                last_episode = self.agent.buffer.buffer[-1]

                states = list(map(lambda ep:ep.state, last_episode.experiences))
                action_probs = self.agent.calculate_action_probs(states, probs=True)
                per_action = list(zip(*action_probs))

                for idx in range(self.agent.env.action_space.n):
                    var_name = "action/" + str(idx)
                    value = np.mean(per_action[idx])
                    self.writer.add_scalar(var_name, value, self.episode_count)


class ContinuousPGLogger(WinLogger):

    def log_episode(self):
        super(ContinuousPGLogger, self).log_episode()

        # level 5 - debugging
        if self.log_level > 4 :

            #  plot action values to facilitate debugging
            if len(self.agent.transitions) > 0:
                states = list(map(lambda exp:exp[0], self.agent.transitions))
                # convert that to actions given the current policy
                action_values = self.agent.calculate_action_values(states, return_values=True)
                num_actions = len(action_values[0])
                # plot for all actions
                for idx in range(num_actions):
                    var_name = "action/" + str(idx)
                    var_value = np.mean(action_values[:, idx])
                    self.writer.add_scalar(var_name, var_value, self.episode_count)


class A2CLogger(WinLogger):

    def log_episode(self):
        super(A2CLogger, self).log_episode()

        # level 5 - debugging
        if self.log_level > 4 :

            # log gradient information
            grad_l2 = np.sqrt(np.mean(np.square(self.agent.grads)))
            grad_max = np.max(np.abs(self.agent.grads))
            grad_var = np.var(self.agent.grads)
            self.writer.add_scalar("grad/l2", grad_l2, self.episode_count)
            self.writer.add_scalar("grad/max", grad_max, self.episode_count)
            self.writer.add_scalar("grad/var", grad_var, self.episode_count)
            
            # log the losses
            self.writer.add_scalar("loss/policy", self.agent.loss_policy_v, self.episode_count)
            self.writer.add_scalar("loss/value", self.agent.loss_value_v, self.episode_count)
            self.writer.add_scalar("loss/entropy", self.agent.loss_entropy_v, self.episode_count)
            loss_total = self.agent.loss_policy_v + self.agent.loss_value_v + self.agent.loss_entropy_v
            self.writer.add_scalar("loss/total", loss_total, self.episode_count)

            # log kl divergence
            self.writer.add_scalar("kl_divergence", self.agent.kl_div_v, self.episode_count)

            # log advantage
            self.writer.add_scalar("advantage", self.agent.mean_adv, self.episode_count)

            # report action probabilities to follow-up how policy evolves
            per_action = list(zip(*self.agent.action_probs))
            for idx in range(self.agent.env.action_space.n):
                var_name = "action/" + str(idx)
                var_value = np.mean(per_action[idx])
                self.writer.add_scalar(var_name, var_value, self.episode_count)


""" 
to log in future implementations

to initialize later
##  All the variables I keep for logging purposes
# intermediate variables
# self.step_reward = None
# self.latest_qvals = None
# self.mean_reward = None
# self.std_reward = None

# initialize history - will coment out for now
# self.hist_rewards = []
# self.hist_steps = [] 


neural network
step
self.writer.add_scalar("q_value/min", min(self.latest_qvals), self.frame_count)
self.writer.add_scalar("q_value/max", max(self.latest_qvals), self.frame_count)
self.writer.add_scalar("loss", loss, self.frame_count)

episode
# monitor parameter sharing
if self.params["SHARING"]:
    self.writer.add_scalar("experiences_received", self.exp_buffer.experiences_received, self.episode_count)
# wont monitor mean and std reward just for now
self.writer.add_scalar("reward_mean", self.mean_reward, self.episode_count)
self.writer.add_scalar("reward_std", self.std_reward, self.episode_count)
track weights
if track_weights:
    self.writer.add_histogram("net_weights", self.gen_params_debug(self.net))
#     self.writer.add_histogram("tgt_net_weights", self.gen_params_debug(self.tgt_net))

no using params, unpack them at initialization


"""

"""

    # def log_episode(self):
    #     super(WinLogger, self).log_episode()

    #     if self.log_level > 2:

    #         # unpack the qtable for debugging
    #         all_values = []
    #         for state, actions in self.agent.qtable.items():
    #             for action, action_value in actions.items():
    #                 all_values.append(action_value)

    #         # import pdb;pdb.set_trace()

    #         self.writer.add_histogram("q-values", np.array(all_values), self.episode_count)

    #     if self.log_level > 1 :

    #         print(self.agent.selected_actions)

"""
