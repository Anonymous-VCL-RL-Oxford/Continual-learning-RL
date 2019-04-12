# Continual Learning Reinforcement Learning environment
I developed a task wrapper that takes as an argument a OpenAI Gym environment, e.g.  Cartpole-v1, and generates new environments with different environment parameters.  For instance in Cartpole-v1,the gravity of the environment as well as the length of the pole and magnetic force can be changed overtasks.  For instance, one can chose to study the impact of changes in environment dynamics, by comparing the performance of the agent when it is on the Earth (gravity of 9.8m.s−2) or the Moon (gravity of 1.62m.s−2)and for different pole lengths (1m and 10m)

The evaluation of the model is done in the same manner than in the Variational Continual Learning paper, i.e.  at task t we train the agent on the environment t until it solved this environment (i.e.  the average score over 100 episodes is 195), then we evaluate the model on previous tasks. In  practice,  evaluation  of  RL  environments  can  be  tricky  because  of  high  variance  because  of  the  initialization  of  the  environment  but  mostly  because  the  last  training  iteration  might  have  badly  updated  the weights.  Therefore, I evaluate at the end of each training episode the score the agent gets on each previous environment (I average the score over 25 runs to reduce variance).  The evaluation of the agent on theithenvironment (i belonging to [[1,t]]), is the average score over the last 50 episodes.

**A PyTorch implementation of both a DQN and a VCL agent are provided. As for now only the DQN agents works fine. To run the DQN agent on the continual learning setting, check Continual-learning-RL.ipynb**

	
## Results
Notice that I limited the episodes the agent can use to learn on a environment t to be less than 1000.  Here I use a simple DQN agent.  The first plot  represents the accuracy (averaged over 25 runs with the DQN agent acting greedily) as a function of the episode.  
These are really interesting results!  Let’s see how we can interpret them. 


1. The DQN agent learns very fast the 2nd task:  my guess is that it is easier to control the pole on the Moon than on the Earth because of lower gravity.  It only take about two hundred episodes to solve the game.
2.  When going to the 3rd task, we can see that the agent struggles to learn:  this is probably because the policy to learn depends a lot on the length of pole (because it impacts the force applied to it).
3.  Same than 1.  here:  the DQN agent learns faster on the Moon.

![](/results/pictures/result_train_v1.png)

1.  The DQN agent did not unlearn task 1 after learning the 2nd.
2.  The agent unlearned the first two tasks after learning on the 3rd (even though it is still way better than random which is around 13).  Furthermore,  since the agent did not fully learn on the 3rd task (because of early stopping), the performance is only about 140.
3.  The agent successfully learn the last task.  By doing so, it enables to increase the performance of all previous tasks.  This is probably due to the fact that the early stopping had a very bad impact onprevious tasks.  In particular it enabled the agent to be quite good at the 3rd task (around 175).

![](/results/pictures/result_test_v1.png)


The VCL agent still has too be fixed. The exploration seems not to work.

![](/results/pictures/VDQN/vdqn_200steps_learning.png)
![](/results/pictures/VDQN/vdqn_200steps_meanvar.png)

![](/results/pictures/VDQN/vdqn_1000steps_learning.png)
![](/results/pictures/VDQN/vdqn_1000steps_meanvar.png)

