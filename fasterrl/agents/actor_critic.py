from fasterrl.agents.policy_gradient import Reinforce, MonteCarloReinforce
from fasterrl.common.network import *

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class A2C(MonteCarloReinforce):

    def set_environment(self, env):
        super(A2C, self).set_environment(env)

        # change eps
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, eps=1e-3)


    def calculate_action_probs(self, states, probs=False):

        states_v = torch.FloatTensor(states)
        logits_v, qvalues_v = self.net(states_v)
        if probs:
            action_probs_v = nn.Softmax(dim=1)(logits_v)
            # v is a tensor which track gradients as well, unpack to acess underlying data
            action_probs = action_probs_v.data.numpy()
            return action_probs

        return logits_v, qvalues_v

    def calculate_kl_div(self, states, prob_v):

        states_v = torch.FloatTensor(states)
        new_logits_v, _ = self.net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()

        return kl_div_v

    def calculate_loss_and_optimize(self, states, actions, values):

        # do the learning
        self.optimizer.zero_grad() # reset gradients

        actions_v = torch.LongTensor(actions)
        values_v = torch.FloatTensor(values)

        logits_v, qvalues_v = self.calculate_action_probs(states, probs=False)

        # save prob_v to calculate KL divergence later
        prob_v = F.softmax(logits_v, dim=1)

        # calculate loss for value
        self.loss_value_v = F.mse_loss(values_v.squeeze(-1), qvalues_v)
        # calculate loss for policy
        log_prob_v = F.log_softmax(logits_v, dim=1)
        # calculate advantage
        # detach, no gradients - qvalues needs to get first
        adv_v = qvalues_v.detach()[0] - values_v.detach()
        # multiply by advantage instead of value
        log_prob_values_v = adv_v * log_prob_v[range(len(actions)), actions]
        self.loss_policy_v = -log_prob_values_v.mean()

        loss_v = self.loss_value_v

        # if use entropy bonus, do additional calculation
        if self.entropy_bonus:
            prob_v = F.softmax(logits_v, dim=1)
            self.loss_entropy_v = self.calculate_entropy_loss(prob_v, log_prob_v)
            loss_v += self.loss_entropy_v

        # propagate policy loss separately t track the gradients
        self.loss_policy_v.backward(retain_graph=True)
        # extract gradient from policy,to plot
        self.grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
            for p in self.net.parameters() if p.grad is not None])

        # propagate remaining loss
        loss_v.backward()

        if self.gradient_clipping:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)
        self.optimizer.step() # change weights

        # metrics for logger
        self.kl_div_v = self.calculate_kl_div(states, prob_v)
        self.action_probs = self.calculate_action_probs(states, probs=True)
        self.mean_adv = torch.mean(adv_v)

        # needs grads and kl_div_v to be accessible in the logger
        # I will start with the easy way of saving this as a class variable
        # and then improving it later

# procedures are correct
# to debug, would have to introduce those other things discussed
# gradient statistics
# KL divergence and other things

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



