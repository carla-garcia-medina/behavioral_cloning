'''
Imports
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gnwrapper

import gym
from gym.wrappers import Monitor

import tensorflow_hub as hub

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

from stable_baselines.gail import generate_expert_traj
from functools import partial
import h5py
import os

from bc_dataset import BCDataset

'''
Create Env and Train Expert
'''
env = gym.make('CartPole-v1')

ppo1_expert = PPO1(MlpPolicy, env, verbose=0)
ppo1_expert.learn(total_timesteps=250000)
ppo1_expert.save("ppo1_cartpole")
ppo1_expert = PPO1.load("ppo1_cartpole")

'''
Define BC Model as NN with a single hidden layer
'''
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Net, self).__init__()
        self.module = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
    	return self.module(x)

'''
Configure the number of trajectory rollouts used to generate the training
and validation datasets, and initialize the corresponding dataset objects
'''
num_episodes = {'train': 10, 'validation': 5}
datasets = {phase: BCDataset(env, num_episodes[phase], ppo1_expert) for phase in ['train', 'validation']}

'''
Set hyperparams
'''
pos_weight = (1 - datasets['train'].prop_pos_exmpls)/datasets['train'].prop_pos_exmpls # use the proportion of pos examples to compute a 'pos_weight' used to normalize binary cross entropy loss used to train the model
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # initialize the criterion function, Binary Cross Entropy Loss w/ Sigmoid
obs_dim = env.observation_space.shape[0] # unpack dimension of the observation space from the gym env
action_dim = env.action_space.shape # unpack dimension of the action space from the gym env
hidden_dim = 32 # set the hidden dimension of the model
learning_rate = 3e-4 # set the learning rate
batch_size = 64 # set the batch size
num_epochs = 50 # set the number of training/validation epochs

'''
Train BC Model
'''
model = Net(obs_dim, action_dim, hidden_dim) # initialize the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # initialize the optimizer used for model trianing
custom_collate_fn = partial(collate_fn, # initialize the collate function via the train dataset stats
                     				obs_mean=datasets['train'].mean_expert_obs,
										 				obs_std=datasets['train'].std_expert_obs) 
dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], # initialize the train dataloader
                                             			batch_size=batch_size, 
																						 			shuffle=True, 
																						 			collate_fn=custom_collate_fn),
							'validation': torch.utils.data.DataLoader(datasets['validation'], # initialize the validation dataloader
                                             			batch_size=len(datasets['validation']), # Q: why do I set the batch size to the whole dataset here + omit shuffling?
																									collate_fn=custom_collate_fn)}
# Training/Validation Loop
for epoch in range(num_epochs+1):
	avg_accuracy = {}
	for phase in ['train', 'validation']:
		total_incorrect = 0
		for batch in iter(dataloaders[phase]):
			expert_obs, expert_actions = batch
			pred_actions = model(expert_obs)
			if phase == 'train':
				optimizer.zero_grad(set_to_none=True) # Q: why is it important to zero the optimizer gradients before computing the loss?
				loss = criterion(pred_actions, expert_actions)
				loss.backward()
				optimizer.step() # To do: look at the torch documention to make sure you understand what happens under the hood when you call 'step' on any optimizer
			else:
				with torch.no_grad(): # Q: why do we explicitly ~not~ track gradients during validation?
					loss = criterion(pred_actions, expert_actions)
		 	# Our version of cartpole features discrete policies, which output actions 0 or 1 at each timestep.
			# When training our BC model, however, we don't explicitly discretize the model outputs (Q: why?). 
			# To calculate the current model accuracy per batch, we do need to discretize the model outputs, 
			# casting any output action strictly greater than 0.5 to 1 and and less than or equal to 0.5 to 0.
			total_incorrect += torch.sum(torch.abs((torch.clip(pred_actions, 0, 1) + 0.5).int() - expert_actions))
	 	# Compute the average accuracy over each phase
		avg_accuracy[phase] = (len(datasets[phase]) - total_incorrect)/len(datasets[phase])
	# Print training/validation stats every 25 epochs
	if epoch % 25 == 0:
		print(f'epoch %i | avg train accuracy: %.2f%% | avg valid accuracy: %.2f%%' % (epoch, 
																																			np.array(avg_accuracy['train']).mean() * 100, 
																																			np.array(avg_accuracy['validation']).mean() * 100))
'''
Render BC Agent and Generate Gifs
'''
env = gnwrapper.LoopAnimation(gym.make('CartPole-v1')) # Start Xvfb

obs = env.reset()

while True:
  env.render()
  # Q: why do we need to process the observations here when rollout out the model?
  processed_obs = (torch.as_tensor(obs) - datasets['train'].mean_expert_obs)/datasets['train'].std_expert_obs
  with torch.no_grad():
    processed_action = (model(processed_obs.float()) + 0.5).int().numpy()[0]
  obs, rewards, done, info = env.step(processed_action)
  if done: 
      break;

env.display()