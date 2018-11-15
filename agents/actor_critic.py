from .policy_gradient import Reinforce, MonteCarloReinforce
from fasterRL.common.network import *

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class A2C(MonteCarloReinforce):

    # already has network with two outputs
    # first is the policy, second is the value 

    # what changes? 
    # just in how I calculate q values
    # and how I calculate loss and optimize

    def calculate_action_probs(self, states, probs=False):

        states_v = torch.FloatTensor(states) 
        logits_v, qvalues_v = self.net(states_v)
        if probs:
            action_probs_v = nn.Softmax(dim=1)(logits_v)
            # v is a tensor which track gradients as well, unpack to acess underlying data            
            action_probs = action_probs_v.data.numpy()
            return action_probs

        return logits_v, qvalues_v

    def calculate_loss_and_optimize(self, states, actions, values):

        # do the learning
        self.optimizer.zero_grad() # reset gradients

        actions_v = torch.LongTensor(actions)
        values_v = torch.FloatTensor(values)

        logits_v, qvalues_v = self.calculate_action_probs(states, probs=False)

        # calculate loss for value
        loss_value_v = F.mse_loss(values_v.squeeze(-1), qvalues_v)

        # calculate loss for policy
        log_prob_v = F.log_softmax(logits_v, dim=1)
        # calculate advantage
        adv_v = qvalues_v - values_v.detach()
        # multiply by advantage instead of value
        log_prob_values_v = adv_v * log_prob_v[range(len(actions)), actions]
        loss_policy_v = -log_prob_values_v.mean()

        loss_v = loss_value_v + loss_policy_v

        # if use entropy bonus, do additional calculation
        if self.entropy_bonus:
            prob_v = F.softmax(logits_v, dim=1)
            entropy_loss_v = self.calculate_entropy_loss(prob_v, log_prob_v)
            loss_v += entropy_loss_v

        loss_v.backward() # propagate gradients

        if self.gradient_clipping:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)           

        self.optimizer.step() # change weights

    

# procedures are correct
# to debug, I would have to introduce those other things discussed
# gradient statistics
# KL divergence and other things

# tomorrow, start introducing these extra things and debugging A2C. goal is to get it to work. then move on to DDPG
# stil, if I have the time, I do want to do n-steps or lambda when possible
# it will get me thinking
# discretization is also a goal



"""
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

                # do the learning
                self.optimizer.zero_grad() # reset gradients

                states, actions, _ = zip(*self.all_transitions)
                self.calculate_loss_and_optimize(states, actions, self.all_values)

                # reset all variables
                self.all_transitions = []
                self.all_values = []
                self.episodes = 0

"""



