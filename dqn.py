import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agents.q_learner import Q_learner


def dqn(agent=None, n_episodes=10000, max_t=200, eps_start=1, eps_end=0.01, eps_decay=0.995):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 195.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoints/checkpoint.pth')
            break
    return scores

args = dict()
args["BUFFER_SIZE"] = int(500)  # replay buffer size
args["BATCH_SIZE"] = 32  # minibatch size
args["GAMMA"] = 1  # discount factor
args["TAU"] = 1e-3  # for soft update of target parameters
args["LR"] = 0.001  # learning rate
args["UPDATE_EVERY"] = 4  # how often to update the network

print("---------------------Starting experiment---------------------")
env = gym.make('CartPole-v1')
env.seed(0)
agent1 = Q_learner(state_size=env.observation_space.shape[0], action_size= env.action_space.n, seed=0, hiddens = [100,100], args = args)
#agent2 = Q_learner(state_size=env.observation_space.shape[0], action_size= env.action_space.n, seed=0, hiddens = [100,100], args = args)

scores1 = dqn(agent1)
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores1)), scores1)

plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title("DQN")
plt.show()