import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim,  hidden_dims):
        super(ActorNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.tanh(layer(x))
        x = self.output_layer(x)
        x = torch.tanh(x)
        return x


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim,  hidden_dims):
        super(CriticNet, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            nn.Linear(state_dim + action_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        for layer in self.hidden_layers:
            x = self.tanh(layer(x))
        x = self.output_layer(x)
        return x


class Tot_CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim,  hidden_dims):
        super(Tot_CriticNet, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            nn.Linear(state_dim + action_dim + action_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state, action1, action2):
        x = torch.cat([state, action1, action2], dim=1)
        for layer in self.hidden_layers:
            x = self.tanh(layer(x))
        x = self.output_layer(x)
        return x


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], gamma=0.99, tau=0.005, Q_lr=1e-3, pi_lr=3e-4, memory_size=500000, device='cpu'):
        self.device = torch.device(device)
        self.actor = ActorNet(state_dim, action_dim,
                              hidden_dims).to(self.device)
        self.critic = CriticNet(state_dim, action_dim,
                                hidden_dims).to(self.device)
        self.target_actor = ActorNet(
            state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = CriticNet(
            state_dim, action_dim, hidden_dims).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.gamma = gamma
        self.tau = tau
        self.Q_lr = Q_lr
        self.pi_lr = pi_lr
        self.memory = deque(maxlen=memory_size)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.pi_lr, weight_decay=1e-8)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.Q_lr, weight_decay=0)
        self.batch_size = 64
        self.policy_frequency = 2
        self.target_frequency = 2

    def act(self, state, epsilon=0):
        with torch.no_grad():
            if np.random.random() < epsilon:
                action = torch.rand(size=(2,)) * 2 - 1
            else:
                state = state.unsqueeze(0)
                action = self.actor(state).squeeze(0)
            return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn_critic(self, transitions):

        state_batch = torch.stack(
            [torch.Tensor(t[0]).to(self.device) for t in transitions])
        action_batch = torch.stack(
            [torch.Tensor(t[1]).to(self.device) for t in transitions])
        reward_batch = torch.stack(
            [torch.Tensor([t[2]]).to(self.device) for t in transitions])
        next_state_batch = torch.stack(
            [torch.Tensor(t[3]).to(self.device) for t in transitions])
        done_batch = torch.stack(
            [torch.Tensor([t[4]]).to(self.device) for t in transitions])

        # Update critic
        with torch.no_grad():
            target_actions = self.target_actor(next_state_batch)
            target_Q = self.target_critic(next_state_batch, target_actions)
            target_Q = reward_batch + self.gamma * target_Q * (1 - done_batch)

        Q = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss, 0

    def learn_actor(self, transitions):
        state_batch = torch.stack(
            [torch.Tensor(t[0]).to(self.device) for t in transitions])

        # Update actor
        actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss, 0

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data)


class DDPGLagrange(DDPG):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], gamma=0.99, tau=0.005,  Q_lr=1e-3, pi_lr=3e-4, memory_size=500000, lagrange_multiplier=0.1, device='cpu'):

        super(DDPGLagrange, self).__init__(
            state_dim, action_dim, hidden_dims, gamma, tau, Q_lr, pi_lr, memory_size, device)
        self.device = torch.device(device)
        self.lagrange_multiplier = lagrange_multiplier
        self.cost_critic = CriticNet(
            state_dim, action_dim, hidden_dims).to(self.device)
        self.target_cost_critic = CriticNet(
            state_dim, action_dim, hidden_dims).to(self.device)
        self.target_cost_critic.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_optimizer = optim.Adam(
            self.cost_critic.parameters(), lr=self.Q_lr, weight_decay=0)

    def remember(self, state, action, reward, cost, next_state, done):
        self.memory.append((state, action, reward, cost, next_state, done))

    def learn_critic(self, transitions):
        state_batch = torch.stack(
            [torch.Tensor(t[0]).to(self.device) for t in transitions])
        action_batch = torch.stack(
            [torch.Tensor(t[1]).to(self.device) for t in transitions])
        reward_batch = torch.stack(
            [torch.Tensor([t[2]]).to(self.device) for t in transitions])
        cost_batch = torch.stack(
            [torch.Tensor([t[3]]).to(self.device) for t in transitions])
        next_state_batch = torch.stack(
            [torch.Tensor(t[4]).to(self.device) for t in transitions])
        done_batch = torch.stack(
            [torch.Tensor([t[5]]).to(self.device) for t in transitions])

        # Update critic
        with torch.no_grad():
            target_actions = self.target_actor(next_state_batch)
            target_Q = self.target_critic(next_state_batch, target_actions)
            target_Q = reward_batch + self.gamma * target_Q * (1 - done_batch)
            target_Qc = self.target_cost_critic(
                next_state_batch, target_actions)
            target_Qc = cost_batch + self.gamma * target_Qc * (1 - done_batch)

        Q = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(Q, target_Q)
        Qc = self.cost_critic(state_batch, action_batch)
        cost_critic_loss = nn.MSELoss()(Qc, target_Qc)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        return critic_loss, cost_critic_loss

    def learn_actor(self, transitions):

        state_batch = torch.stack(
            [torch.Tensor(t[0]).to(self.device) for t in transitions])
        actions = self.actor(state_batch)
        actor_loss = (-self.critic(state_batch, actions) +
                      self.lagrange_multiplier*self.cost_critic(state_batch, actions)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss, 0


class GCDDPG(DDPG):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], gamma=0.99, tau=0.005, Q_lr=1e-3, pi_lr=3e-4, memory_size=500000, device='cpu'):
        super(GCDDPG, self).__init__(
            state_dim, action_dim, hidden_dims, gamma, tau, Q_lr, pi_lr, memory_size, device)
        self.device = torch.device(device)

        half_hidden_dims = [dim // 2 for dim in hidden_dims]
        # half_hidden_dims = hidden_dims
        self.critic = Tot_CriticNet(
            state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = Tot_CriticNet(
            state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.Q_lr, weight_decay=0)

        self.cost_critic = Tot_CriticNet(
            state_dim, action_dim, hidden_dims).to(self.device)
        self.target_cost_critic = Tot_CriticNet(
            state_dim, action_dim, hidden_dims).to(self.device)
        self.target_cost_critic.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_optimizer = optim.Adam(
            self.cost_critic.parameters(), lr=self.Q_lr, weight_decay=0)

        self.actor = ActorNet(state_dim, action_dim,
                              half_hidden_dims).to(self.device)
        self.target_actor = ActorNet(
            state_dim, action_dim, half_hidden_dims).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.pi_lr, weight_decay=1e-8)

        self.cost_actor = ActorNet(
            state_dim, action_dim, half_hidden_dims).to(self.device)
        self.target_cost_actor = ActorNet(
            state_dim, action_dim, half_hidden_dims).to(self.device)
        self.target_cost_actor.load_state_dict(self.cost_actor.state_dict())
        self.cost_actor_optimizer = optim.Adam(
            self.cost_actor.parameters(), lr=self.pi_lr, weight_decay=1e-8)

    def remember(self, state, action1, action2, reward, cost, next_state, done):
        self.memory.append(
            (state, action1, action2, reward, cost, next_state, done))

    def act(self, state, epsilon=0):
        with torch.no_grad():
            if np.random.random() < epsilon:
                action1 = torch.rand(size=(2,)) * 2 - 1
                action2 = torch.rand(size=(2,)) * 2 - 1
            else:
                state = state.unsqueeze(0)
                action1 = self.actor(state).squeeze(0)
                action2 = self.cost_actor(state).squeeze(0)
            return action1, action2

    def learn_critic(self, transitions):

        state_batch = torch.stack(
            [torch.Tensor(t[0]).to(self.device) for t in transitions])
        action1_batch = torch.stack(
            [torch.Tensor(t[1]).to(self.device) for t in transitions])
        action2_batch = torch.stack(
            [torch.Tensor(t[2]).to(self.device) for t in transitions])
        reward_batch = torch.stack(
            [torch.Tensor([t[3]]).to(self.device) for t in transitions])
        cost_batch = torch.stack(
            [torch.Tensor([t[4]]).to(self.device) for t in transitions])
        next_state_batch = torch.stack(
            [torch.Tensor(t[5]).to(self.device) for t in transitions])
        done_batch = torch.stack(
            [torch.Tensor([t[6]]).to(self.device) for t in transitions])

        with torch.no_grad():
            target_actions1 = self.target_actor(next_state_batch)
            target_actions2 = self.target_cost_actor(next_state_batch)
            target_Q = self.target_critic(
                next_state_batch, target_actions1, target_actions2)
            target_Q = reward_batch + self.gamma * target_Q * (1 - done_batch)
            target_Qc = self.target_cost_critic(
                next_state_batch, target_actions1, target_actions2)
            target_Qc = cost_batch + self.gamma * target_Qc * (1 - done_batch)
        Q = self.critic(state_batch, action1_batch, action2_batch)
        critic_loss = nn.MSELoss()(Q, target_Q)
        Qc = self.cost_critic(state_batch, action1_batch, action2_batch)
        cost_critic_loss = nn.MSELoss()(Qc, target_Qc)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()
        return critic_loss, cost_critic_loss

    def learn_actor(self, transitions):

        state_batch = torch.stack(
            [torch.Tensor(t[0]).to(self.device) for t in transitions])

        actions1 = self.actor(state_batch).clone()
        actions2 = self.cost_actor(state_batch).clone()
        target_actions1 = self.target_actor(state_batch)
        target_cost_actions2 = self.target_cost_actor(state_batch)

        actor_loss = -self.critic(state_batch, actions1,
                                  target_cost_actions2).mean()
        
        # cost_actor_loss = self.cost_critic(state_batch, target_actions1, actions2).mean()
        cost_actor_loss = (self.cost_critic(state_batch, target_actions1, actions2) + 2e-23 * torch.exp(
            -(torch.nn.functional.cosine_similarity(target_actions1 + 1e-2 * torch.randn_like(target_actions1), actions2, dim=1)) * 50)).mean()
        # A minor punishment for action in the opposite direction, see ./plot_result/barrier.py
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.cost_actor_optimizer.zero_grad()
        cost_actor_loss.backward()
        self.cost_actor_optimizer.step()
        return actor_loss, cost_actor_loss
