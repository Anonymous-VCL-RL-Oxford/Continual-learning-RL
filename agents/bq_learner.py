import numpy as np
import random
from collections import namedtuple, deque

from models.bayesian_network import Bayesian_QNetwork
from utils.ReplayMemory import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    from torchviz import make_dot, make_dot_from_trace
except ImportError:
    print("Torchviz was not found.")



class BQ_learner():

    def __init__(self, state_size, action_size, hiddens, args, seed, prev_means = None):
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
        self.print_graph_bol = False
        self.task_idx = 0

        self.qnetwork_local = Bayesian_QNetwork(input_size = state_size, output_size = action_size, hidden_size = hiddens, seed = seed, prev_means = prev_means)
        self.qnetwork_target =  Bayesian_QNetwork(input_size = state_size, output_size = action_size, hidden_size = hiddens, seed = seed, prev_means = prev_means)
        self.optimizer = optim.Adam(self.qnetwork_local.weights, lr=self.LR)


        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        self.t_step = 0


    def next_task(self):

        self.qnetwork_local.update_prior()
        self.qnetwork_target.update_prior()
        self.qnetwork_local.create_head()
        self.qnetwork_target.create_head()

        self.full_update()


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                mean_variance = self.learn(experiences, self.GAMMA)
                return mean_variance
    def act(self, state, task_idx = 0, eps=0.):
        self.task_idx = task_idx
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state, no_samples = self.qnetwork_local.no_samples_test, task_idx  = task_idx)

        if random.random() > eps:
            act = np.argmax(action_values.cpu().data.numpy())
            return act
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions = actions.long()
        ##TODO: sample different parameters for each element from the batch
        no_samples_Q_target = 1
        Q_targets_next = self.qnetwork_target.forward(next_states, task_idx = self.task_idx, no_samples=no_samples_Q_target).view(-1,2).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards.repeat([no_samples_Q_target,1]) + (gamma * Q_targets_next * (1 - dones.repeat([no_samples_Q_target,1])))
        """
        #unparallelized


        no_samples_Q_target = 1
        Q_targets_next = self.qnetwork_target.forward_diff_params(next_states, no_samples=1).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
"""
        loss = self.qnetwork_local.get_loss(states, actions, Q_targets, no_samples_Q_target)
        if self.print_graph_bol:
            # Just if you want to see the computational graph
             # mf_model.get_loss(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device), task_id), params=params)
            print_graph(self.qnetwork_local, loss)
            self.print_graph_bol = False

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

        mean_variance = self.qnetwork_local.get_variance()
        return mean_variance


    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.weights, local_model.weights):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def full_update(self):
        for target_param, local_param in zip(self.qnetwork_target.weights, self.qnetwork_local.weights):
            target_param.data.copy_( local_param.data )


def print_graph(model, output):
    params = dict()
    for i in range(len(model.W_m)):
        params["W_m{}".format(i)] = model.W_m[i]
        params["W_v{}".format(i)] = model.W_v[i]
        params["b_m{}".format(i)] = model.b_m[i]
        params["b_v{}".format(i)] = model.b_v[i]
        params["prior_W_m".format(i)] = model.prior_W_m[i]
        params["prior_W_v".format(i)] = model.prior_W_v[i]
        params["prior_b_m".format(i)] = model.prior_b_m[i]
        params["prior_b_v".format(i)] = model.prior_b_v[i]

    for i in range(len(model.W_last_m)):
         params["W_last_m".format(i)] = model.W_last_m[i]
         params["W_last_v".format(i)] = model.W_last_v[i]
         params["b_last_m".format(i)] = model.b_last_m[i]
         params["b_last_v".format(i)] = model.b_last_v[i]
         params["prior_W_last_m".format(i)] = model.prior_W_last_m[i]
         params["prior_W_last_v".format(i)] = model.prior_W_last_v[i]
         params["prior_b_last_m".format(i)] = model.prior_b_last_m[i]
         params["prior_b_last_v".format(i)] = model.prior_b_last_v[i]
    dot = make_dot(output, params=params)
    dot.view()

    return

