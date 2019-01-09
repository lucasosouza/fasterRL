"""
Creates neural networks 
"""

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class Network(nn.Module):

    def __init__(self, device="cpu", random_seed=42):
        torch.manual_seed(random_seed)
        super(Network, self).__init__()

        # self.device = device

    def forward(self, x):
        """ Main forward function """      
        
        return self.network(x)

class ConvNetwork(Network):

    def __init__(self, device="cpu", random_seed=42):
        super(ConvNetwork, self).__init__(device, random_seed)

        # to replace by specific 
        self.conv = None
        self.fc = None

    def _get_conv_out(self, shape):
        """ 
            Get shape of output of conv layers, to help defining the input shape of next fc layer
        """
        # pass a batch with 1 obs and shape equal input shape through conv layers
        o = self.conv(torch.zeros(1, *shape))
        output_shape = int(np.prod(o.size()))

        return output_shape

    def forward(self, x):
        """ Main forward function """      

        # apply the convolution layer to input and obtain a 4d tensor on output
        # and result is flattened, by the view function
        # view doesn't create a new memory obect or move data in memort, 
        # just change higher-level shape of tensor
        conv_out = self.conv(x).view(x.size()[0], -1)       

        # pass flattened 2d tensor to fc layer
        return self.fc(conv_out)     

class DeepQNetwork(ConvNetwork):

    # input for Pong is 210 x 160
    # these were planned for an 84x84 image
    # maybe what I can do is use the wrapper to rebalance it 
    
    def __init__(self, input_shape, n_actions, device="cpu", random_seed=42):
        super(DeepQNetwork, self).__init__(device, random_seed)

        # defines convolutional layers as defined in DQN paper
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()        
        )
        
        # get size of last layer to pass it to the first fc layer
        # since pytorch has no flatten layer
        conv_out_size = self._get_conv_out(input_shape)
        
        # defines fully connected layers as defined in DQN paper
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
class SimpleValueNetwork(Network):
    
    def __init__(self, input_shape, n_actions, device="cpu", random_seed=42):
        super(SimpleValueNetwork, self).__init__(device, random_seed)

        # simple network for average case
        # when one dimensional, needs to extract first variable
        self.network = nn.Sequential(
            nn.Linear(input_shape[0], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_actions)
        )

class SimplePolicyNetwork(Network):

    def __init__(self, input_shape, n_actions, device="cpu", random_seed=42):
        super(SimplePolicyNetwork, self).__init__(device, random_seed)

        # simple network for average case
        # when one dimensional, needs to extract first variable
        self.network = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # need to think about when to apply the softmax 
        # maybe the methods should go here in this network instead of someplace else


class ContinuousPolicyNetwork(Network):

    def __init__(self, input_shape, action_space, device="cpu", random_seed=42):
        super(ContinuousPolicyNetwork, self).__init__(device, random_seed)

        # define action boundaries to clip network output
        self.n_actions = action_space.shape[0]
        action_lower_bounds = action_space.low
        action_upper_bounds = action_space.high

        self.action_lower_bounds = torch.FloatTensor(action_lower_bounds).to(device)
        action_range = action_upper_bounds - action_lower_bounds
        self.action_mult_factor = torch.FloatTensor(action_range / 2.).to(device) # divided by range of tanh 

    def adjust_output_range(self, x): 
        """ Adjust output to the action range expected in the environment
            Do I let it calculate gradients on this? yes, I would think so
        """ 

        # converts output of tanh to 0 to 2 range
        x = x + 1.
        # multiply by the range in env divided by range in neural network
        x = x * self.action_mult_factor
        # add lower bound of env range
        x = x + self.action_lower_bounds

        return x

    def forward(self, x):
        x = super(ContinuousPolicyNetwork, self).forward(x)

        return self.adjust_output_range(x)

class SimpleContinuousPolicyNetwork(ContinuousPolicyNetwork):

    def __init__(self, input_shape, actions_shape, action_lower_bounds,action_range, device="cpu", random_seed=42):
        super(SimpleContinuousPolicyNetwork, self).__init__(input_shape, action_space, device, random_seed)

        hidden_layer_neurons = 128

        # simple network for average case
        # when one dimensional, needs to extract first variable
        self.base = nn.Sequential(
            nn.Linear(input_shape[0], hidden_layer_neurons),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(hidden_layer_neurons, self.n_actions),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(hidden_layer_neurons, self.n_actions),
            nn.Softplus()
        )
        
    def forward(self, x):

        x = self.base(x)

        return self.adjust_output_range(self.mu(x)), self.var(x)


class SimpleA2CNetwork(Network):

    def __init__(self, input_shape, n_actions, device="cpu", random_seed=42):
        super(SimpleA2CNetwork, self).__init__(device, random_seed)

        hidden_layer_neurons = 128

        self.base = nn.Sequential(
            nn.Linear(input_shape[0], hidden_layer_neurons),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_layer_neurons, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_layer_neurons, 1)
        )

    def forward(self, x):

        x = self.base(x)

        return self.policy(x), self.value(x)


class DDPGActor(ContinuousPolicyNetwork):

    def __init__(self, input_shape, action_space, device="cpu", random_seed=42):

        super(DDPGActor, self).__init__(input_shape, action_space, device, random_seed)

        n_vars_actions = action_space.shape[0]
        n_vars_state = input_shape[0]
 
        self.network = nn.Sequential(
           nn.Linear(n_vars_state, 400), nn.ReLU(),
           nn.Linear(400, 300), nn.ReLU(),
           nn.Linear(300, n_vars_actions), nn.Tanh()
        )

class DDPGCritic(Network):

    def __init__(self, input_shape, action_shape, device="cpu", random_seed=42):
        super(DDPGCritic, self).__init__(device, random_seed)

        n_vars_actions = action_shape[0]
        n_vars_state = input_shape[0]

        self.obs_network = nn.Sequential(
            nn.Linear(n_vars_state, 400), nn.ReLU()
        )

        self.out_network = nn.Sequential(
            nn.Linear(400 + n_vars_actions, 300), nn.ReLU(),
            nn.Linear(300,1)
        )

    def forward(self, x, actions):

        obs = self.obs_network(x)
        out = self.out_network(torch.cat([obs, actions], dim=1))

        return out

