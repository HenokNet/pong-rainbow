import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], [], []
        
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        return indices, samples, weights
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-5
        self.max_priority = max(self.max_priority, priorities.max())
    
    def update_beta(self):
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def __len__(self):
        return len(self.buffer)