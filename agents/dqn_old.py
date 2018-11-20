from .base_agent import ValueBasedAgent
from fasterRL.common.network import *
from fasterRL.common.buffer import Experience, ExperienceBuffer

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class DQN(ValueBasedAgent):

    def __init__(self, params):
        super(DQN, self).__init__(params)

        experience_buffer_size = 1000
        if "EXPERIENCE_BUFFER_SIZE" in params:
            experience_buffer_size = params["EXPERIENCE_BUFFER_SIZE"]

        self.replay_batch_size = 32
        if "REPLAY_BATCH_SIZE" in params:
            self.replay_batch_size = params["REPLAY_BATCH_SIZE"]

        # type of network
        self.network_type = SimpleValueNetwork
        if "NETWORK_TYPE" in params:
            self.network_type = eval(params["NETWORK_TYPE"])

        # gradient clipping
        self.gradient_clipping = False
        if "GRADIENT_CLIPPING" in params:
            self.grad_l2_clip = 1
            if "GRAD_L2_CLIP" in params:
                self.grad_l2_clip  = self.params["GRAD_L2_CLIP"]

        # double q learning
        self.double_qlearning = False
        if "DOUBLE_QLEARNING" in params:
            self.double_qlearning = params["DOUBLE_QLEARNING"]

        # soft or hard update of target network
        self.soft_update = False
        if "SOFT_UPDATE" in params:
            self.soft_update = params["SOFT_UPDATE"]
            if self.soft_update:
                self.soft_update = True
                self.soft_update_tau = 5e-3
                if "SOFT_UPDATE_TAU" in params:
                    self.soft_update_tau = params["SOFT_UPDATE_TAU"]
        else: 
            self.sync_target_frames = 2000
            self.frame_count = 0
            if "SYNC_TARGET_FRAMES" in params:
                self.sync_target_frames = params["SYNC_TARGET_FRAMES"]

        # initialize experience buffer
        self.buffer = ExperienceBuffer(experience_buffer_size)
 
    def set_environment(self, env):
        super(DQN, self).set_environment(env)

        # initialize networks
        self.net = self.network_type(env.observation_space.shape, env.action_space.n, 
            device=self.device, random_seed=self.random_seed)
        self.tgt_net = self.network_type(env.observation_space.shape, env.action_space.n, 
            device=self.device, random_seed=self.random_seed)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def select_best_action(self, state):

        q_vals_v = self.calculate_q_vals(state)
        # chooses greedy action and get its value
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())

        return action

    def calculate_q_vals(self, state=None):

        # if called from external program
        if state is None:
            state = self.state

        # moves state into an array with 1 sample to pass through neural net
        state_a = np.array([state], copy=False)
        # creates tensor
        state_v = torch.tensor(state_a).to(self.device)
        # get q values with feed forward
        q_vals_v = self.net(state_v)

        return q_vals_v  

    def learn(self, action, next_state, reward, done):

        # append experience to buffer
        self.buffer.append(Experience(self.state, action, reward, done, next_state))

        ## learn when there are enough batch samples
        ## ideally I should accumulate a mass of experiences before starting to learn
        if len(self.buffer) > self.replay_batch_size:
            # zero gradients
            self.optimizer.zero_grad()
            # sample from buffer
            batch = self.buffer.sample(self.replay_batch_size)
            # calculate loss
            loss_t = self.calc_loss(batch)
            # calculate gradients
            loss_t.backward()
            # gradient clipping
            if self.gradient_clipping:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_l2_clip)
            # optimize
            self.optimizer.step()

    def update_params(self):
        super(DQN, self).update_params()

        # merge network and target network according to specified strategy
        if self.soft_update:
            self.soft_update_target_network()
        else:
            self.frame_count += 1
            if self.frame_count == self.sync_target_frames:
                self.hard_update_target_network()
                self.frame_count = 0

    def hard_update_target_network(self):
        """ Update every X steps """

        self.tgt_net.load_state_dict(self.net.state_dict())

    def soft_update_target_network(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """        
        
        # iterate through both together and make a copy one by one
        for target_param, local_param in zip(self.tgt_net.parameters(), self.net.parameters()):
            target_param.data.copy_(
                self.soft_update_tau*local_param.data + (1-self.soft_update_tau)*target_param.data
            )

    def unpack_batch(self, batch):

        states, actions, rewards, dones, next_states = batch
        
        # creates tensors. and push them to device, if GPU is available, then uses GPU
        states_v = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)

        return states_v, next_states_v, rewards_v, actions_v, done_mask

    def calc_loss(self, batch):
        """ Function optimized to exploit GPU parallelism by processing all batch samples with vector operations """

        # unpack vectors of variables
        states_v, next_states_v, rewards_v, actions_v, done_mask = self.unpack_batch(batch)

        # calculate state-action values
        # gather: select only the values for the actions taken
        # result of gather applied to tensor is differentiable operation, keep all gradients w.r.t to final loss value
        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                        
        # apply target network to next_states. get maximum q-value. no need to know the action, just the value
        # if double, get actions from regular network and value from target network
        # avoids overfitting
        if self.double_qlearning:
            # get q values with feed forward
            next_q_vals_v = self.net(next_states_v)
            # chooses greedy action from target net
            _, next_state_action_v = torch.max(next_q_vals_v, dim=1)
            # gets actions from 
            next_state_values = \
                self.tgt_net(next_states_v).gather(1, next_state_action_v.unsqueeze(-1)).squeeze(-1)
        else:
            next_state_values = self.tgt_net(next_states_v).max(1)[0]
 
        # if is done, value of next state is set to 0. important correction
        next_state_values[done_mask] = 0.0
 
        # detach values from computation graph, to prevent gradients from flowing into neural net
        next_state_values = next_state_values.detach()
        
        # calculate total value (Bellman approximation value)
        expected_state_action_values = next_state_values * self.gamma + rewards_v
        
        # calculate mean squared error loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        
        return loss

"""
TODO:
see if I can move everythin related to torch to network 
to make it agnostic, if possible
want to make a test replacing it for a tensorflow network 
this should include the soft and hard updates, and the gradient clipping

"""

