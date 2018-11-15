from .base_agent import BaseAgent
from fasterRL.common.buffer import ShortExperience, EpisodeBuffer
from fasterRL.common.network import *

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class PolicyGradient(BaseAgent):

    def __init__(self, params, alias="agent"):
        super(PolicyGradient, self).__init__(params, alias)

        # type of network
        self.network_type = SimplePolicyNetwork
        if "NETWORK_TYPE" in params:
            self.network_type = eval(params["NETWORK_TYPE"])


    def set_environment(self, env):
        super(PolicyGradient, self).set_environment(env)

        self.net = self.network_type(env.observation_space.shape, env.action_space.n, 
            device=self.device, random_seed=self.random_seed)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def calculate_action_probs(self, states, probs=False):

        states_v = torch.FloatTensor(states) 
        logits_v = self.net(states_v)
        if probs:
            action_probs_v = nn.Softmax(dim=1)(logits_v)
            # v is a tensor which track gradients as well, unpack to acess underlying data            
            action_probs = action_probs_v.data.numpy()
            return action_probs

        return logits_v

    def select_action(self):

        action_probs = self.calculate_action_probs([self.state], probs=True)[0]
        action = np.random.choice(len(action_probs), p=action_probs)

        return action


class CrossEntropy(PolicyGradient):

    def __init__(self, params, alias="agent"):
        super(CrossEntropy, self).__init__(params, alias)

        cutoff_percentile = 70
        if "CUTOFF_PERCENTILE" in params:
            cutoff_percentile = params["CUTOFF_PERCENTILE"]

        episode_buffer_size = 16
        if "EPISODE_BUFFER_SIZE" in params:
            episode_buffer_size = params["EPISODE_BUFFER_SIZE"]

        # initialize episode buffer
        self.buffer = EpisodeBuffer(episode_buffer_size, cutoff_percentile)

    def reset(self):
        super(CrossEntropy, self).reset()
        self.episode_reward = 0.0

    def learn(self, action, next_state, reward, done):

        self.episode_reward += reward
        self.buffer.append_experience(ShortExperience(self.state, action))

        if done:
            buffer_full = self.buffer.append_episode(self.episode_reward)
            # only learns when buffer is full
            if buffer_full: 
                states, actions = self.buffer.sample()
                logits_v = self.calculate_action_probs(states, probs=False)
                actions_v = torch.LongTensor(actions)
                self.optimizer.zero_grad() # reset gradients
                loss_v = nn.CrossEntropyLoss()(logits_v, actions_v) # calculate loss
                loss_v.backward() # propagate gradients
                self.optimizer.step() # change weights

class Reinforce(PolicyGradient):

    def __init__(self, params, alias="agent"):
        super(Reinforce, self).__init__(params, alias)

        # baseline
        self.baseline_qvalue = False
        if "BASELINE_QVALUE" in params:
            self.baseline_qvalue = params["BASELINE_QVALUE"]

        # entropy bonus
        self.entropy_bonus = False
        if "ENTROPY_BONUS" in params:
            self.entropy_bonus = params["ENTROPY_BONUS"]
            if self.entropy_bonus:
                self.entropy_beta = 0.01
                if "ENTROPY_BETA" in params:
                    self.entropy_beta = params["ENTROPY_BETA"]

        # gradient clipping
        self.gradient_clipping = False
        if "GRADIENT_CLIPPING" in params:
            self.gradient_clipping = params["GRADIENT_CLIPPING"]
            if self.gradient_clipping:
                self.clip_grad = 0.1
                if "CLIP_GRAD" in params:
                    self.clip_grad = params["CLIP_GRAD"]

    def calculate_entropy_loss(self, prob_v, log_prob_v):

        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = - self.entropy_beta * entropy_v
        return entropy_loss_v

    def calculate_qvalues(self):

        values = []
        value = 0
        for state, action, reward in reversed(self.transitions):
            value = reward + self.gamma * value
            values.append(value)

        values = values[::-1] # revert back to original order of transitions

        return values

    def calculate_loss_and_optimize(self, states, actions, values):

        # do the learning
        self.optimizer.zero_grad() # reset gradients

        actions_v = torch.LongTensor(actions)
        values_v = torch.FloatTensor(values)

        logits_v = self.calculate_action_probs(states, probs=False)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_values_v = values_v * log_prob_v[range(len(actions)), actions]
        loss_v = -log_prob_values_v.mean()

        # if use entropy bonus, do additional calculation
        if self.entropy_bonus:
            prob_v = F.softmax(logits_v, dim=1)
            entropy_loss_v = self.calculate_entropy_loss(prob_v, log_prob_v)
            loss_v += entropy_loss_v

        loss_v.backward() # propagate gradients
        self.optimizer.step() # change weights


class MonteCarloReinforce(Reinforce):

    def reset(self):
        super(Reinforce, self).reset()

        self.transitions = []

    def learn(self, action, next_state, reward, done):
  
        self.transitions.append((self.state, action, reward))

        if done:        

            values = self.calculate_qvalues()
            # remove baseline
            if self.baseline_qvalue:
                values -= np.mean(values)

            states, actions, _ = zip(*self.transitions)
            self.calculate_loss_and_optimize(states, actions, values)


class BatchReinforce(Reinforce):
    # implement a crude version to test, no buffer, then improve if ok

    def __init__(self, params, alias="agent"):
        super(BatchReinforce, self).__init__(params, alias)

        self.episode_buffer_size = 16
        if "EPISODE_BUFFER_SIZE" in params:
            self.episode_buffer_size = params["EPISODE_BUFFER_SIZE"]

        self.transitions = []
        self.all_values = []
        self.all_transitions = []
        self.episodes = 0

    def learn(self, action, next_state, reward, done):
  
        self.transitions.append((self.state, action, reward))

        if done:        

            # calculate values
            values = self.calculate_qvalues()

            # keep track of batch. note: move to buffer later
            self.all_values.extend(values)
            self.all_transitions.extend(self.transitions)
            self.transitions = []
            self.episodes += 1

            if self.episodes == self.episode_buffer_size:

                # remove baseline (a baseline for the batch)
                if self.baseline_qvalue:
                    self.all_values -= np.mean(self.all_values)


                # do learning
                states, actions, _ = zip(*self.all_transitions)
                self.calculate_loss_and_optimize(states, actions, self.all_values)

                # reset all variables
                self.all_transitions = []
                self.all_values = []
                self.episodes = 0


""""


# similar to MonteCarlo
# won't need to account
# but need to store transitions as well
# which can be seen as experiences if I use the same logic of the buffer
# but I don't need to use the same logic if I don't want to - since I'm not keeping them, I don't really need a buffer - it is just MonteCarlo
# is it every visit or first visit? seems like Every Visit
# let's rock and roll and test both


opted to do an average of the episode instead of keeping a list of past rewards
and doing a moving average
would introduce a new hyperparameter dependent on the environment, number of rewards to keep track of in the moving average
or if I used a stepsize approach to update the average reward, a learning rate / step size for the average reward. anyway it would be an additional parameter I don't want to keep track of 

        # baseline
        self.use_baseline = False
        if "BASELINE" in params:
            if params["BASELINE"]:
                self.use_baseline = True
                baseline_ma_period = 1000
                if "BASELINE_MA_PERIOD" in params:
                    baseline_ma_period = params["BASELINE_MA_PERIOD"]
                self.last_values = deque(maxlen=baseline_ma_period)


"""