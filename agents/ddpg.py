# ddpg.py
from .base_agent import ValueBasedAgent
from fasterRL.common.network import *
from fasterRL.common.buffer import Experience, ExperienceBuffer
from fasterRL.common.exploration import OUNoise

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDPG(ValueBasedAgent):

    def __init__(self, params):
        super(DDPG, self).__init__(params)

        experience_buffer_size = 1000
        if "EXPERIENCE_BUFFER_SIZE" in params:
            experience_buffer_size = params["EXPERIENCE_BUFFER_SIZE"]

        self.replay_batch_size = 32
        if "REPLAY_BATCH_SIZE" in params:
            self.replay_batch_size = params["REPLAY_BATCH_SIZE"]

        # gradient clipping
        self.gradient_clipping = False
        if "GRADIENT_CLIPPING" in params:
            self.grad_l2_clip = 1
            if "GRAD_L2_CLIP" in params:
                self.grad_l2_clip  = self.params["GRAD_L2_CLIP"]

        # removed double-qlearning
        # see howit can be added later

        # params regarding OU exploration
        self.ou_exploration = True
        if "OU_EXPLORATION" in params:
            self.ou_exploration = params["OU_EXPLORATION"]
            if self.ou_exploration:
                ou_mu = 0.0
                if "OU_MU" in params:
                    ou_mu  = self.params["OU_MU"]
                ou_theta = 0.15
                if "OU_THETA" in params:
                    ou_theta  = self.params["OU_THETA"]
                ou_sigma = 0.2
                if "OU_SIGMA" in params:
                    ou_sigma  = self.params["OU_SIGMA"]
                # epsilon can be used to slowly decay exploration over time
                # currently not doing decay
                self.ou_epsilon = 1.0
                if "OU_EPSILON" in params:
                    self.ou_epsilon  = self.params["OU_EPSILON"]

                # initialize ounoise class
                self.ou_noise = OUNoise(ou_mu, ou_theta, ou_sigma)

        # soft or hard update of target network
        # for now, keep single parameter for both
        self.soft_update = True
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

        self.env = env
        self.reset()

        # need to save boundaries to clip after adding the noise
        self.action_lower_bounds = self.env.action_space.low
        self.action_upper_bounds = self.env.action_space.high

        # initialize networks
        self.net_actor = DDPGActor(env.observation_space.shape, env.action_space, 
            device=self.device, random_seed=self.random_seed)
        self.tgtnet_actor = DDPGActor(env.observation_space.shape, env.action_space, 
            device=self.device, random_seed=self.random_seed)

        self.net_critic = DDPGCritic(env.observation_space.shape, env.action_space.shape, 
            device=self.device, random_seed=self.random_seed)
        self.tgtnet_critic = DDPGCritic(env.observation_space.shape, env.action_space.shape, 
            device=self.device, random_seed=self.random_seed)

        self.actor_optimizer = optim.Adam(self.net_actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.net_critic.parameters(), lr=self.learning_rate)

        # what about complex state environments? need to treat this in a different way later
        self.ou_noise.set_size(self.env.action_space.shape[0])


    def reset(self):
        super(DDPG, self).reset()

        # zero ou_noise state as well
        self.ou_noise.reset()


    def select_action(self):

        # get deterministic action values
        states_v = torch.FloatTensor([self.state])         
        action_values = self.net_actor(states_v).data.cpu().numpy()[0]

        #  add the noise
        if self.ou_exploration and self.ou_epsilon > 0:
            noise = self.ou_noise.sample()
            action_values += self.ou_epsilon * noise

        # clip to expected range
        action_values = np.clip(action_values, self.action_lower_bounds, self.action_upper_bounds)

        return action_values

    def unpack_batch(self, batch):

        states, actions, rewards, dones, next_states = batch
        
        # creates tensors. and push them to device, if GPU is available, then uses GPU
        states_v = torch.FloatTensor(states).to(self.device)
        next_states_v = torch.FloatTensor(next_states).to(self.device)
        rewards_v = torch.FloatTensor(rewards).to(self.device)
        actions_v = torch.FloatTensor(actions).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)

        return states_v, next_states_v, rewards_v, actions_v, done_mask

    def learn(self, action, next_state, reward, done):

        # append experience to buffer
        self.buffer.append(Experience(self.state, action, reward, done, next_state))

        ## learn when there are enough batch samples
        ## ideally I should accumulate a mass of experiences before starting to learn
        if len(self.buffer) > self.replay_batch_size:
            # sample from buffer
            batch = self.buffer.sample(self.replay_batch_size)
            # unpack vectors of variables
            states_v, next_states_v, rewards_v, actions_v, done_mask = self.unpack_batch(batch)

            #### train critic
            # zero gradients
            self.critic_optimizer.zero_grad()

            # get q_values
            qvalues_v = self.net_critic(states_v, actions_v)

            # get the next actions based on tgt net and their values
            next_actions_v = self.tgtnet_actor(next_states_v)
            next_qvalues_v = self.tgtnet_critic(next_states_v, next_actions_v)
            # make next values 0 if episode is done
            next_qvalues_v[done_mask] = 0.0 

            # calculated expected state action values
            expected_qvalues_v = rewards_v.unsqueeze(dim=-1) + next_qvalues_v * self.gamma

            # with expected, can now calculate loss
            critic_loss_v = F.mse_loss(qvalues_v, expected_qvalues_v)

            # backpropagates and update
            critic_loss_v.backward()
            self.critic_optimizer.step()

            #### train actor
            self.actor_optimizer.zero_grad()

            # expected actions
            current_actions_v = self.net_actor(states_v)

            # calculate loss
            # negated output of the critic is loss to backpropate the action network
            actor_loss_v = (- self.net_critic(states_v, current_actions_v)).mean()

            # backpropagates and update
            actor_loss_v.backward()
            self.actor_optimizer.step()

    #### Methods related to target network update

    def update_params(self):
        super(DDPG, self).update_params()

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

        self.tgtnet_actor.load_state_dict(self.net_actor.state_dict())
        self.tgtnet_critic.load_state_dict(self.net_critic.state_dict())

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
        # for actor
        for target_param, local_param in zip(self.tgtnet_actor.parameters(), self.net_actor.parameters()):
            target_param.data.copy_(
                self.soft_update_tau*local_param.data + (1-self.soft_update_tau)*target_param.data
            )

        # for critic
        for target_param, local_param in zip(self.tgtnet_critic.parameters(), self.net_critic.parameters()):
            target_param.data.copy_(
                self.soft_update_tau*local_param.data + (1-self.soft_update_tau)*target_param.data
            )


