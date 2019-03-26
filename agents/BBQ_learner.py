import numpy as np
import random
from collections import namedtuple, deque

from models.bayesian_network import Bayesian_QNetwork

from utils.ReplayMemory import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

#To check graph
#from torchviz import make_dot, make_dot_from_trace
#dot = make_dot(loss, params = dict(self.qnetwork_local.named_parameters()))
#dot.view()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BBQ_learner():

    def __init__(self, state_size, action_size,  hiddens, args, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        self.hiddens = hiddens
        self.BUFFER_SIZE = args["BUFFER_SIZE"]
        self.BATCH_SIZE = args["BATCH_SIZE"]
        self.GAMMA = args["GAMMA"]
        self.UPDATE_EVERY = args["UPDATE_EVERY"]
        self.LR = args["LR"]
        self.TAU = args["TAU"]
        self.PI, self.SIGMA_1, self.SIGMA_2 = args["PI"], args["SIGMA_1"], args["SIGMA_2"]

        self.qnetwork_local = Bayesian_QNetwork(state_size, action_size , hiddens, seed).to(device)
        self.qnetwork_target = Bayesian_QNetwork(state_size, action_size , hiddens, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)[0][0]
        self.qnetwork_local.eval()
        with torch.no_grad():
            z_dist = torch.from_numpy(
                np.array([[self.Vmin + i * self.delta_z for i in range(self.N)]])).to(device)
            z_dist = torch.unsqueeze(z_dist, 2).float()
            Q_dist, _  = self.qnetwork_local(state)#
            Q_dist = Q_dist.detach()
            Q_dist = Q_dist#.reshape(-1,  self.action_size, self.N)
            Q_target = torch.matmul(Q_dist, z_dist).squeeze(1)
            a_star = torch.argmax(Q_target, dim=1)[0]

        if eps != 0.:
            self.qnetwork_local.train()

        if random.random() > eps:
            return a_star.cpu().data.numpy()[0]
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        max_Q = self.qnetwork_target.forward(next_states)
        targets = rewards + (1 - dones) * self.GAMMA * max_Q
        predictions = self.qnetwork_local.forward(next_states)
        loss = - torch.mean((targets - predictions) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
