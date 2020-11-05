from torch import nn
import torch

class Policy_NN(nn.Module):
    def __init__(self):
        super(Policy_NN, self).__init__()
        self.state_space = 6#env.get_state_space().shape[0]
        self.action_space = 2#env.get_action_space().shape[0]

        self.l1 = nn.Linear(self.state_space, 2)#have it smaller
        self.l2 = nn.Linear(2, self.action_space)

    def forward(self, x):
        x = self.l1(x)
        x = nn.ReLU()(x)
        x = self.l2(x)
        x = nn.Softmax(dim=-1)(x)
        return x

    def num_param(self):
        return count_parameters(self)

class Policy_Param(nn.Module):
    def __init__(self):
        super(Policy_Param, self).__init__()
        self.state_space = 6 #env.get_state_space().shape[0]

        self.l1 = nn.Linear(self.state_space, 1, bias = False)# each param here works as the prob.
        self.l1.weight.data = (torch.randn(6)/10+0.7).float() # initialize the param in [0,1)
    def forward(self, x):
        x = self.l1(x)
        x = nn.ReLU()(x) # in case there's any negative value
        return x

    def num_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
