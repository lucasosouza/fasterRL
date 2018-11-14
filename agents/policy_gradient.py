from .base_agent import BaseAgent
from fasterRL.common.buffer import ShortExperience, EpisodeBuffer
from fasterRL.common.network import *

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class MonteCarloReinforce(PolicyGradient):

    def reset(self):
        super(Reinforce, self).reset()

        self.transitions = []

    def learn(self, action, next_state, reward, done):
  
        self.transitions.append((self.state, action, reward))

        if done:        

            # calculate values
            values = []
            value = 0
            for state, action, reward in reversed(self.transitions):
                value = reward + self.gamma * value
                values.append(value)
            values = values[::-1] # revert back to original order of transitions

            # do the learning
            self.optimizer.zero_grad() # reset gradients

            states, actions, _ = zip(*self.transitions)
            actions_v = torch.LongTensor(actions)
            values_v = torch.FloatTensor(values)

            logits_v = self.calculate_action_probs(states, probs=False)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_values_v = values_v * log_prob_v[range(len(actions)), actions]
            loss_v = -log_prob_values_v.mean()

            loss_v.backward() # propagate gradients
            self.optimizer.step() # change weights


class BatchReinforce(PolicyGradient):
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
            values = []
            value = 0
            for state, action, reward in reversed(self.transitions):
                value = reward + self.gamma * value
                values.append(value)
            values = values[::-1] # revert back to original order of transitions

            self.all_values.extend(values)
            self.all_transitions.extend(self.transitions)
            self.transitions = []
            self.episodes += 1

            if self.episodes == self.episode_buffer_size:

                # do the learning
                self.optimizer.zero_grad() # reset gradients

                states, actions, _ = zip(*self.all_transitions)
                actions_v = torch.LongTensor(actions)
                values_v = torch.FloatTensor(self.all_values)

                logits_v = self.calculate_action_probs(states, probs=False)
                log_prob_v = F.log_softmax(logits_v, dim=1)
                log_prob_values_v = values_v * log_prob_v[range(len(actions)), actions]
                loss_v = -log_prob_values_v.mean()

                loss_v.backward() # propagate gradients
                self.optimizer.step() # change weights

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

"""