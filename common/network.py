"""
Creates neural networks 
"""

import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, device="cpu", random_seed=42):
        torch.manual_seed(random_seed)
        super(Network, self).__init__()

        self.device = device

    def forward(self, x):
        """ Main forward function """      
        
        # manually changed from torch.FloatTensor to torch.cuda.FloatTensor to run in GPU
        if self.device == "cuda":
            x = x.type(torch.cuda.FloatTensor)
        elif self.device == "cpu":
            x = x.type(torch.FloatTensor)    

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
        
        # manually changed from torch.FloatTensor to torch.cuda.FloatTensor to run in GPU
        if self.device == "cuda":
            x = x.type(torch.cuda.FloatTensor)
        elif self.device == "cpu":
            x = x.type(torch.FloatTensor)
        
        # apply the convolution layer to input and obtain a 4d tensor on output
        # and result is flattened, by the view function
        # view doesn't create a new memory obect or move data in memort, 
        # just change higher-level shape of tensor
        conv_out = self.conv(x).view(x.size()[0], -1)       

        # pass flattened 2d tensor to fc layer
        return self.fc(conv_out)     

class DeepQNetwork(ConvNetwork):
    
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
