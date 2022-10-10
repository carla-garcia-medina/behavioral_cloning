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
from torch.utils.tensorboard import SummaryWriter


env = gym.make('CartPole-v1')

ppo1_expert = PPO1(MlpPolicy, env, verbose=0)
ppo1_expert.learn(total_timesteps=250000)
ppo1_expert.save("ppo1_cartpole")