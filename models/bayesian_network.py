###THIS CODE HAS BEEN MOSTLY RETRIEVED ON THE GITHUB OF NITARSHAN"
#https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb

import torch
from torch import nn
import torch.nn.functional as F
import math
import torch.optim as optim
from scipy.stats import truncnorm
from copy import deepcopy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
truncated = False

def truncated_normal(size, stddev=1, variable = False, mean=0):
    if not truncated:
        stdv = 1/size[-1]
        X_tensor = torch.ones(size).uniform_(-stdv, stdv).to(device=device)
        X_tensor.requires_grad = variable
    else:
        mu, sigma = mean, stddev
        lower, upper= -2 * sigma, 2 * sigma
        X = truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        X_tensor = torch.Tensor(data = X.rvs(size)).to(device = device)
        X_tensor.requires_grad = variable
    return X_tensor

def init_tensor(value,  dout, din = 1, variable = False):
    if din != 1:
        x = value * torch.ones([din, dout]).to(device = device)
    else:
        x = value * torch.ones([dout]).to(device = device)
    x.requires_grad=variable

    return x

class Cla_NN(object):
    def __init__(self, input_size, hidden_size, output_size):
        return

class Bayesian_QNetwork(Cla_NN):
    def __init__(self, input_size, output_size, hidden_size,  seed = 0, no_samples=10, single_head = True, prev_means = None, learning_rate=0.001):
        ##TODO: handle single head
        super(Bayesian_QNetwork, self).__init__(input_size, hidden_size, output_size)

        m1, v1, hidden_size = self.create_weights(
             input_size, hidden_size, output_size, prev_means)
        self.no_samples = no_samples
        self.no_samples_train = 1
        self.no_samples_test = 1
        self.input_size = input_size
        self.out_size = output_size
        self.size = hidden_size
        self.single_head = single_head

        self.W_m, self.b_m = m1[0], m1[1]
        self.W_v, self.b_v = v1[0], v1[1]

        self.W_last_m, self.b_last_m = [], []
        self.W_last_v, self.b_last_v = [], []


        m2, v2 = self.create_prior(input_size, self.size, output_size)

        self.prior_W_m, self.prior_b_m, = m2[0], m2[1]
        self.prior_W_v, self.prior_b_v = v2[0], v2[1]

        self.prior_W_last_m, self.prior_b_last_m = [], []
        self.prior_W_last_v, self.prior_b_last_v = [], []

        self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
        self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
        self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
        self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None


        self.no_layers = len(self.size) - 1
        self.learning_rate = learning_rate

        if prev_means is not None:
            self.init_first_head(prev_means)
        else:
            self.create_head()


        m1.append(self.W_last_m)
        m1.append(self.b_last_m)
        v1.append(self.W_last_v)
        v1.append(self.b_last_v)

        r1 = m1 + v1
        self.weights = [item for sublist in r1 for item in sublist]


        #self.optimizer = optim.Adam(self.weights, lr=learning_rate)
        return


    def _prediction(self, inputs, task_idx = 0, no_samples = None, noise = True):
        if no_samples == None:
            K = self.no_samples
        else:
            K = no_samples
        size = self.size

        act = torch.unsqueeze(inputs, 0).repeat([K, 1, 1])
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
            eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)
            if noise:
                weights = torch.add(eps_w * torch.exp(0.5*self.W_v[i]), self.W_m[i])
                biases = torch.add(eps_b * torch.exp(0.5*self.b_v[i]), self.b_m[i])
            else:
                weights = self.W_m[i].unsqueeze(0)
                biases =  self.b_m[i].unsqueeze(0)

            pre = torch.add(torch.einsum('mni,mio->mno', act, weights), biases)
            act = F.relu(pre)

        din = self.size[-2]
        dout = self.size[-1]

        eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
        eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)
        Wtask_m = self.W_last_m[task_idx]
        Wtask_v = self.W_last_v[task_idx]
        btask_m = self.b_last_m[task_idx]
        btask_v = self.b_last_v[task_idx]


        if noise:
            weights = torch.add(eps_w * torch.exp(0.5 * Wtask_v), Wtask_m)
            biases = torch.add(eps_b * torch.exp(0.5 * btask_v), btask_m[i])
        else:
            weights = Wtask_m.unsqueeze(0)
            biases = btask_m.unsqueeze(0)

        act = torch.unsqueeze(act, 3)
        weights = torch.unsqueeze(weights, 1)
        pre = torch.add(torch.sum(act * weights, dim = 2), biases)
        return pre

    def forward(self, inputs, noise = True, task_idx = 0, no_samples = None):
        pred = self._prediction(inputs, task_idx, no_samples, noise).mean(0)
        return pred

    def _logpred_regression(self, inputs, actions, targets, task_idx):
        pred = self._prediction(inputs, task_idx, self.no_samples_train).view(-1, self.out_size).gather(1, actions.repeat([self.no_samples_train, 1]))
        targets = targets.repeat([self.no_samples_train, 1])
        log_lik = - torch.mean((pred- targets)**2)
        return log_lik


    def _logpred_classification(self, inputs, targets, task_idx):
        """not needed in the RL case (regression)"""
        loss = torch.nn.CrossEntropyLoss()
        pred = self._prediction(inputs, task_idx, self.no_samples).view(-1,self.out_size)
        targets = targets.repeat([self.no_samples, 1]).view(-1)
        log_liks = -loss(pred, targets.type(torch.long))
        log_lik = log_liks.mean()
        return log_lik


    def _KL_term(self):
        kl = 0
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            m, v = self.W_m[i], self.W_v[i]
            m0, v0 = self.prior_W_m[i], self.prior_W_v[i]

            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_m[i], self.b_v[i]
            m0, v0 = self.prior_b_m[i], self.prior_b_v[i]

            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl +=  log_std_diff + mu_diff_term + const_term

        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]

        for i in range(no_tasks):
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]

            const_term = - 0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]

            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term
        return kl

    def get_loss(self, batch_x, actions, batch_y, task_idx =0):
        loss = - self._logpred_regression(batch_x, actions, batch_y, task_idx)
        #loss += 2e-2 * self._KL_term()/batch_x.shape[0]

        return loss


    def save_weights(self):
        ''' Save weights before training on the coreset before getting the test accuracy '''

        print("Saving weights before core set training")
        self.W_m_copy = [self.W_m[i].clone().detach().data for i in range(len(self.W_m))]
        self.W_v_copy = [self.W_v[i].clone().detach().data for i in range(len(self.W_v))]
        self.b_m_copy = [self.b_m[i].clone().detach().data for i in range(len(self.b_m))]
        self.b_v_copy = [self.b_v[i].clone().detach().data for i in range(len(self.b_v))]

        self.W_last_m_copy = [self.W_last_m[i].clone().detach().data for i in range(len(self.W_last_m))]
        self.W_last_v_copy = [self.W_last_v[i].clone().detach().data for i in range(len(self.W_last_v))]
        self.b_last_m_copy = [self.b_last_m[i].clone().detach().data for i in range(len(self.b_last_m))]
        self.b_last_v_copy = [self.b_last_v[i].clone().detach().data for i in range(len(self.b_last_v))]

        self.prior_W_m_copy = [self.prior_W_m[i].data for i in range(len(self.prior_W_m))]
        self.prior_W_v_copy = [self.prior_W_v[i].data for i in range(len(self.prior_W_v))]
        self.prior_b_m_copy = [self.prior_b_m[i].data for i in range(len(self.prior_b_m))]
        self.prior_b_v_copy = [self.prior_b_v[i].data for i in range(len(self.prior_b_v))]

        self.prior_W_last_m_copy = [self.prior_W_last_m[i].data for i in range(len(self.prior_W_last_m))]
        self.prior_W_last_v_copy = [self.prior_W_last_v[i].data for i in range(len(self.prior_W_last_v))]
        self.prior_b_last_m_copy = [self.prior_b_last_m[i].data for i in range(len(self.prior_b_last_m))]
        self.prior_b_last_v_copy = [self.prior_b_last_v[i].data for i in range(len(self.prior_b_last_v))]

        return

    def load_weights(self):
        ''' Re-load weights after getting the test accuracy '''

        print("Reloading previous weights after core set training")
        self.weights = []
        self.W_m = [self.W_m_copy[i].clone().detach().data for i in range(len(self.W_m))]
        self.W_v = [self.W_v_copy[i].clone().detach().data for i in range(len(self.W_v))]
        self.b_m = [self.b_m_copy[i].clone().detach().data for i in range(len(self.b_m))]
        self.b_v = [self.b_v_copy[i].clone().detach().data for i in range(len(self.b_v))]

        for i in range(len(self.W_m)):
            self.W_m[i].requires_grad = True
            self.W_v[i].requires_grad = True
            self.b_m[i].requires_grad = True
            self.b_v[i].requires_grad = True

        self.weights += self.W_m
        self.weights += self.W_v
        self.weights += self.b_m
        self.weights += self.b_v


        self.W_last_m = [self.W_last_m_copy[i].clone().detach().data for i in range(len(self.W_last_m))]
        self.W_last_v = [self.W_last_v_copy[i].clone().detach().data for i in range(len(self.W_last_v))]
        self.b_last_m = [self.b_last_m_copy[i].clone().detach().data for i in range(len(self.b_last_m))]
        self.b_last_v = [self.b_last_v_copy[i].clone().detach().data for i in range(len(self.b_last_v))]

        for i in range(len(self.W_last_m)):
            self.W_last_m[i].requires_grad = True
            self.W_last_v[i].requires_grad = True
            self.b_last_m[i].requires_grad = True
            self.b_last_v[i].requires_grad = True

        self.weights += self.W_last_m
        self.weights += self.W_last_v
        self.weights += self.b_last_m
        self.weights += self.b_last_v

        self.optimizer = optim.Adam(self.weights, lr=self.learning_rate)
        self.prior_W_m = [self.prior_W_m_copy[i].data for i in range(len(self.prior_W_m))]
        self.prior_W_v = [self.prior_W_v_copy[i].data for i in range(len(self.prior_W_v))]
        self.prior_b_m = [self.prior_b_m_copy[i].data for i in range(len(self.prior_b_m))]
        self.prior_b_v = [self.prior_b_v_copy[i].data for i in range(len(self.prior_b_v))]

        self.prior_W_last_m = [self.prior_W_last_m_copy[i].data for i in range(len(self.prior_W_last_m))]
        self.prior_W_last_v = [self.prior_W_last_v_copy[i].data for i in range(len(self.prior_W_last_v))]
        self.prior_b_last_m = [self.prior_b_last_m_copy[i].data for i in range(len(self.prior_b_last_m))]
        self.prior_b_last_v = [self.prior_b_last_v_copy[i].data for i in range(len(self.prior_b_last_v))]

        return

    def clean_copy_weights(self):
        self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
        self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
        self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
        self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None

    def create_head(self):
        ''''Create new head when a new task is detected'''
        print("creating a new head")
        din = self.size[-2]
        dout = self.size[-1]

        W_m= truncated_normal([din, dout], stddev=0.1, variable=True)
        b_m= truncated_normal([dout], stddev=0.1, variable=True)
        W_v = init_tensor(-6.0,  dout = dout, din = din, variable= True)
        b_v = init_tensor(-6.0,  dout = dout, variable= True)

        self.W_last_m.append(W_m)
        self.W_last_v.append(W_v)
        self.b_last_m.append(b_m)
        self.b_last_v.append(b_v)


        W_m_p = torch.zeros([din, dout]).to(device = device)
        b_m_p = torch.zeros([dout]).to(device = device)
        W_v_p =  init_tensor(1,  dout = dout, din = din)
        b_v_p = init_tensor(1, dout = dout)

        self.prior_W_last_m.append(W_m_p)
        self.prior_W_last_v.append(W_v_p)
        self.prior_b_last_m.append(b_m_p)
        self.prior_b_last_v.append(b_v_p)
        self.weights = []
        self.weights += self.W_m
        self.weights += self.W_v
        self.weights += self.b_m
        self.weights += self.b_v
        self.weights += self.W_last_m
        self.weights += self.W_last_v
        self.weights += self.b_last_m
        self.weights += self.b_last_v
        self.optimizer = optim.Adam(self.weights, lr=self.learning_rate)

        return

    def init_first_head(self, prev_means):
        ''''When the MFVI_NN is instanciated, we initialize weights with those of the Vanilla NN'''
        print("initializing first head")
        din = self.size[-2]
        dout = self.size[-1]
        self.prior_W_last_m = [torch.zeros([din, dout]).to(device = device)]
        self.prior_b_last_m = [torch.zeros([dout]).to(device = device)]
        self.prior_W_last_v =  [init_tensor(1,  dout = dout, din = din)]
        self.prior_b_last_v = [init_tensor(1, dout = dout)]

        W_last_m = prev_means[2][0].detach().data
        W_last_m.requires_grad = True
        self.W_last_m = [W_last_m]
        self.W_last_v = [init_tensor(-6.0,  dout = dout, din = din, variable= True)]


        b_last_m = prev_means[3][0].detach().data
        b_last_m.requires_grad = True
        self.b_last_m = [b_last_m]
        self.b_last_v = [init_tensor(-6.0, dout = dout, variable= True)]

        return

    def create_weights(self, in_dim, hidden_size, out_dim, prev_means):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)

        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []
        W_v = []
        b_v = []

        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]
            if prev_means is not None:
                W_m_i = prev_means[0][i].detach().data
                W_m_i.requires_grad = True
                bi_m_i = prev_means[1][i].detach().data
                bi_m_i.requires_grad = True
            else:
            #Initializiation values of means
                W_m_i= truncated_normal([din, dout], stddev=0.1, variable=True)
                bi_m_i= truncated_normal([dout], stddev=0.1, variable=True)
            #Initializiation values of variances
            W_v_i = init_tensor(-6.0,  dout = dout, din = din, variable = True)
            bi_v_i = init_tensor(-6.0,  dout = dout, variable = True)

            #Append to list weights
            W_m.append(W_m_i)
            b_m.append(bi_m_i)
            W_v.append(W_v_i)
            b_v.append(bi_v_i)

        return [W_m, b_m], [W_v, b_v], hidden_size

    def create_prior(self, in_dim, hidden_size, out_dim, initial_mean = 0, initial_variance = 1):

        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []

        W_v = []
        b_v = []

        for i in range(no_layers - 1):
            din = hidden_size[i]
            dout = hidden_size[i + 1]

            # Initializiation values of means
            W_m_val = initial_mean * torch.zeros([din, dout]).to(device = device)
            bi_m_val = initial_mean * torch.zeros([dout]).to(device = device)

            # Initializiation values of variances
            W_v_val = initial_variance * init_tensor(1,  dout = dout, din = din )
            bi_v_val =  initial_variance * init_tensor(1,  dout = dout)

            # Append to list weights
            W_m.append(W_m_val)
            b_m.append(bi_m_val)
            W_v.append(W_v_val)
            b_v.append(bi_v_val)

        return [W_m, b_m], [W_v, b_v]




    def update_prior(self):
        print("updating prior...")
        for i in range(len(self.W_m)):
            self.prior_W_m[i].data.copy_(self.W_m[i].clone().detach().data)
            self.prior_b_m[i].data.copy_(self.b_m[i].clone().detach().data)
            self.prior_W_v[i].data.copy_(torch.exp(self.W_v[i].clone().detach().data))
            self.prior_b_v[i].data.copy_(torch.exp(self.b_v[i].clone().detach().data))

        length = len(self.W_last_m)

        for i in range(length):
            self.prior_W_last_m[i].data.copy_(self.W_last_m[i].clone().detach().data)
            self.prior_b_last_m[i].data.copy_(self.b_last_m[i].clone().detach().data)
            self.prior_W_last_v[i].data.copy_(torch.exp(self.W_last_v[i].clone().detach().data))
            self.prior_b_last_v[i].data.copy_(torch.exp(self.b_last_v[i].clone().detach().data))

        return