from .base_agent import BaseAgent
from fasterRL.common.buffer import ShortExperience, EpisodeBuffer
from fasterRL.common.network import *

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class CrossEntropy(BaseAgent):

    def __init__(self, params, alias="agent"):
        super(CrossEntropy, self).__init__(params, alias)

        cutoff_percentile = 70
        if "CUTOFF_PERCENTILE" in params:
            cutoff_percentile = params["CUTOFF_PERCENTILE"]

        episode_buffer_size = 16
        if "EPISODE_BUFFER_SIZE" in params:
            episode_buffer_size = params["EPISODE_BUFFER_SIZE"]

        # type of network
        self.network_type = SimplePolicyNetwork
        if "NETWORK_TYPE" in params:
            self.network_type = eval(params["NETWORK_TYPE"])

        # initialize episode buffer
        self.buffer = EpisodeBuffer(episode_buffer_size, cutoff_percentile)

    def reset(self):
        super(CrossEntropy, self).reset()
        self.episode_reward = 0.0

    def set_environment(self, env):
        super(CrossEntropy, self).set_environment(env)

        self.net = self.network_type(env.observation_space.shape, env.action_space.n, 
            device=self.device, random_seed=self.random_seed)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def calculate_action_probs(self, states, probs=False):

        states_v = torch.FloatTensor(states) 
        action_probs_v = self.net(states_v)
        if probs:
            action_probs_v = nn.Softmax(dim=1)(action_probs_v)
            # v is a tensor which track gradients as well, unpack to acess underlying data            
            action_probs = action_probs_v.data.numpy()
            return action_probs

        return action_probs_v

    def learn(self, action, next_state, reward, done):

        self.episode_reward += reward
        self.buffer.append_experience(ShortExperience(self.state, action))

        if done:
            buffer_full = self.buffer.append_episode(self.episode_reward)
            # only learns when buffer is full
            if buffer_full: 
                states, actions = self.buffer.sample()
                action_probs_v = self.calculate_action_probs(states)
                actions_v = torch.LongTensor(actions)
                self.optimizer.zero_grad() # reset gradients
                loss_v = nn.CrossEntropyLoss()(action_probs_v, actions_v) # calculate loss
                loss_v.backward() # propagate gradients
                self.optimizer.step() # change weights

    def select_action(self):

        action_probs = self.calculate_action_probs([self.state], probs=True)[0]
        action = np.random.choice(len(action_probs), p=action_probs)

        return action


    # can I reuse learn? yes I can
    # but I will only learn after the last episode is over and my buffer has reached full capacity
    # I can control it in the buffer
    # when it is done, I will ask to close the buffer