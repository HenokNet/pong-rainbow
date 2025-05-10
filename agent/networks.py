import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class DistributionalDQN(nn.Module):
    def __init__(self, state_size, action_size, atoms=51, v_min=-10, v_max=10):
        super(DistributionalDQN, self).__init__()
        self.action_size = action_size
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.feature = nn.Sequential(
            NoisyLinear(state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            NoisyLinear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.advantage = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, action_size * atoms)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, atoms)
        )
    
    def forward(self, x):
        dist = self.dist(x)
        return torch.sum(dist * self.support, dim=2)
    
    def dist(self, x):
        x = self.feature(x)
        advantage = self.advantage(x).view(-1, self.action_size, self.atoms)
        value = self.value(x).view(-1, 1, self.atoms)
        dist = value + advantage - advantage.mean(1, keepdim=True)
        return F.softmax(dist, dim=-1)
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()