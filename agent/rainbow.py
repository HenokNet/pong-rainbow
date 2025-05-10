import torch
import numpy as np
import random
import math
from collections import deque
from .memory import PrioritizedReplayBuffer
from .networks import DistributionalDQN

class RainbowDQNAgent:
    def __init__(self, state_size=7, action_size=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        
        # Distributional RL
        self.atoms = 51
        self.v_min = -10
        self.v_max = 10
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(self.device)
        
        # Memory
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.n_step = 3
        self.batch_size = 512
        self.target_update = 500
        self.learning_rate = 0.0001
        
        # Networks
        self.model = DistributionalDQN(state_size, action_size, self.atoms, self.v_min, self.v_max).to(self.device)
        self.target_model = DistributionalDQN(state_size, action_size, self.atoms, self.v_min, self.v_max).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Tracking
        self.steps = 0
        self.last_loss = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.random_actions = deque(maxlen=1000)

    def act(self, state, training=False):
        if training and np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
            self.random_actions.append(1)
            return action
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
            self.random_actions.append(0)
            return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        indices, samples, weights = self.memory.sample(self.batch_size)
        states, actions, projected_dist = self._compute_n_step_returns(samples)
        
        dist = self.model.dist(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.atoms)
        dist = dist.gather(1, actions).squeeze(1)
        
        loss = - (projected_dist * torch.log(dist + 1e-8)).sum(1)
        priorities = loss.detach().cpu().abs().numpy() + 1e-5
        loss = (loss * weights.to(self.device)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        self.memory.update_priorities(indices, priorities)
        self.last_loss = loss.item()
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.model.reset_noise()
        self.target_model.reset_noise()
        
        return loss.item()

    def _compute_n_step_returns(self, samples):
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            target_dist = self.target_model.dist(next_states)
            target_dist = target_dist[range(self.batch_size), next_actions]
            projected_dist = self._project_distribution(rewards, dones, target_dist)
        
        return states, actions, projected_dist

    def _project_distribution(self, rewards, dones, target_dist):
        batch_size = rewards.size(0)
        delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        
        projected_dist = torch.zeros((batch_size, self.atoms), device=self.device)
        rewards = rewards.unsqueeze(1).expand_as(projected_dist)
        dones = dones.unsqueeze(1).expand_as(projected_dist)
        support = self.support.unsqueeze(0).expand_as(projected_dist)
        
        Tz = rewards + (1 - dones) * (self.gamma ** self.n_step) * support
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        offset = torch.linspace(0, (batch_size-1)*self.atoms, batch_size).long()\
            .unsqueeze(1).expand(batch_size, self.atoms).to(self.device)
        
        projected_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (target_dist * (u.float() - b)).view(-1))
        projected_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (target_dist * (b - l.float())).view(-1))
        
        return projected_dist