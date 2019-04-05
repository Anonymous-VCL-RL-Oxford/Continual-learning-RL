import numpy as np
import random
from collections import namedtuple, deque

from models.bayesian_network import Bayesian_QNetwork
from utils.ReplayMemory import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BQ_learner():

    def __init__(self, state_size, action_size, hiddens, args, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.hiddens = hiddens
        self.BUFFER_SIZE = args["BUFFER_SIZE"]
        self.BATCH_SIZE = args["BATCH_SIZE"]
        self.GAMMA = args["GAMMA"]
        self.UPDATE_EVERY = args["UPDATE_EVERY"]
        self.LR = args["LR"]
        self.TAU = args["TAU"]


        self.qnetwork_local = Bayesian_QNetwork(state_size, action_size, hiddens, seed)
        self.qnetwork_target = Bayesian_QNetwork(state_size, action_size, hiddens, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.weights, lr=self.LR)

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

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state, no_samples = self.qnetwork_local.no_samples_test)

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions = actions.long()
        ##TODO: sample different parameters for each element from the batch
        no_samples_Q_target = 10
        Q_targets_next = self.qnetwork_target.forward(next_states, no_samples=no_samples_Q_target).view(-1,2).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards.repeat([no_samples_Q_target,1]) + (gamma * Q_targets_next * (1 - dones.repeat([no_samples_Q_target,1])))

        #unparallelized
        """
        no_samples_Q_target = 1
        Q_targets_next = self.qnetwork_target.forward_diff_params(next_states, no_samples=1).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        """
        loss = self.qnetwork_local.get_loss(states, actions, Q_targets, no_samples_Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.weights, local_model.weights):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
