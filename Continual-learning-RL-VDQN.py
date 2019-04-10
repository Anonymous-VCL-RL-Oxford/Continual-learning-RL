#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import torch
import numpy as np
from collections import deque
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from utils.cartpole import CartPoleEnv
from agents.q_learner import Q_learner
from agents.bq_learner import BQ_learner


# In[2]:


args = dict()
args["BUFFER_SIZE"] = int(500)  # replay buffer size
args["BATCH_SIZE"] = 32  # minibatch size
args["GAMMA"] = 0.95  # discount factor
args["TAU"] = 1e-3  # for soft update of target parameters
args["LR"] = 0.001  # learning rate
args["UPDATE_EVERY"] = 4  # how often to update the network

env_name = 'CartPole-v1'

def my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


# In[3]:


def transform_dict_to_tuple(param):
    param_list = []
    if "seed" not in param.keys():
        param_list += [0]
    else:
        param_list += [param["seed"]]
        
    if "length" not in param.keys():
        param_list += [0.5]
    else:
        param_list += [param["length"]]
        
    if "gravity" not in param.keys():
        param_list += [9.8]
    else:
        param_list += [param["gravity"]]
        
    if "force_mag" not in param.keys():
        param_list += [10.0]
    else:
        param_list += [param["force_mag"]]
    return tuple(param_list)


# In[4]:


class Task_Wrapper():
    def __init__(self, env_name, params):
        self.env_name = env_name
        self.params = list(my_product(params))
        self.current_param = 0
        self.seed = seed
        self.envs = []
        
    def next_task(self):
        params = self.params[self.current_param]
        params_tuple = transform_dict_to_tuple(params)
        env = CartPoleEnv(**params)
        env.seed(self.seed)
        self.current_param+=1
        self.envs.append({params_tuple : env})
        return self.envs
    
    def get_env(self, index):
        params = self.params[index]
        env = CartPoleEnv(**params)
        env.seed(self.seed)
        return env 

class Queue():
    def __init__(self, capacity):
        self.capacity = capacity-1
        self.queue = []
        self.nb_elems = -1
        
    def add(self, elem):
        if self.nb_elems == self.capacity:
            self.pop()
            self.add(elem)
        else:
            self.queue.append(elem)
            self.nb_elems+=1
    
    def pop(self):        
        self.nb_elems -=1
        return self.queue.pop(0)


# In[5]:



def dqn(envs, agent = None, n_episodes=20, max_t=200,     eps_start=1, eps_end=0.01, eps_decay=0.995, desactivate_noise = False):
    scores_test = [Queue(50) for i in range(len(envs))]
    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start                    
    env = list(envs[-1].values())[0]
    for i_episode in range(1, n_episodes+1):

        state = env.reset()
        score = 0
        if desactivate_noise:
            eps = 0
            
        for t in range(max_t):
            action = agent.act(state = state,task_idx = len(envs)-1, eps = eps )
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        test_dqns(scores_test, envs, agent)

        score_averaged = scores_test[-1].queue[-1]
        scores_window.append(score_averaged)       
        scores.append(score_averaged)              
        eps = max(eps_end, eps_decay*eps)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0 and i_episode>100:
            break
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            #torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoints/checkpoint.pth')
       
    scores_test_list = [np.array(scores_test[i].queue).mean() for i in range(len(scores_test)) ]
    return scores, scores_test_list


def test_dqns(scores_test, envs, agent, n_episodes = 5, max_t = 1000):
    for i in range(len(envs)):
        env_i = list(envs[i].values())[0]
        scores_test[i].add(test_dqn(env_i, agent, i,  n_episodes, max_t))
            
            
def test_dqn(env, agent, task_idx = 0, n_episodes = 1, max_t=1000):
    _scores = 0                       
    for i_episode in range(1, n_episodes+1):
        _state = env.reset()
        _score = 0
        for t in range(max_t):
            _action = agent.act(_state, task_idx,  0.0)
            _next_state, _reward, _done, _ = env.step(_action)
            _state = _next_state
            _score += _reward
            if _done:
                break 
        _scores +=  _score              
    return _scores/n_episodes


# <h2>Agent definition</h2>

# In[6]:


#agent = Q_learner(state_size=4, action_size=2, seed=0, hiddens = [100,100], args = args)
hiddens = [100,100]



# In[7]:


params = {"length": [1, 10], 
         "gravity": [9.8, 1.62],
          "seed":[0]
         }

print("Params: (Seed, Length, Gravity, Force_mag)")
seed = 0
desactivate_noise = False
task_wrapper = Task_Wrapper(env_name,params)
scores = dict()
test_scores = dict()
for task_id in range(len(task_wrapper.params)):
        
    print("------------ Task n°{}/{} ------------".format(task_id+1,len(task_wrapper.params) ))
    envs = task_wrapper.next_task()
    param_tuple = list(envs[-1].keys())[0]
    print("Current param: {}".format(param_tuple))

    
    if task_id == 0 :
        print("Let's first train a Vanilla network")
        vanilla_agent = Q_learner(state_size=4, action_size=2, seed=0, hiddens = hiddens,
                           args = args)
        _, test_score = dqn(envs, vanilla_agent)
        print(test_score)
        weights = vanilla_agent.get_weights()
        agent = BQ_learner(state_size=4, action_size=2, seed=0, hiddens = hiddens, 
                           args = args, prev_means = weights)
        desactivate_noise = True
        
        print("We can now start using VDQNs")

    scores[param_tuple], test_scores[param_tuple] = dqn(envs, agent, desactivate_noise = desactivate_noise)
    agent.next_task()
    print(test_scores[param_tuple])


# In[ ]:


columns = ["Task#","Seed", "Gravity", "Length", "Force_mag", "Episode", "Score"]
df = pd.DataFrame(columns = columns)
for j,param in enumerate(list(scores.keys())):
    print(j)
    values = scores[param]
    liste = []

    for i in range(len(values)):
        liste.append([j, param[0], param[1], param[2],param[3], i, values[i]])
    df2 = pd.DataFrame(data = liste, columns = columns)
    df = pd.concat([df,df2])
    df.reset_index()
path= "results/vdqn.csv"
df.to_csv(path)


# Test error

# In[ ]:


score_list = [np.array(list(test_scores.keys()))]
for key in test_scores.keys():
   
    score_list.append(np.array(test_scores[key]))
scores= np.array(score_list)
print(scores)
path= "results/vdqn.npy"
np.save(path, scores)


# In[ ]:




