from gym_utils import *
import torch

from pstd.sd.ddpm import DDPMSampler
from stable_baselines3 import PPO
import os
from stable_baselines3.common.policies import obs_as_tensor

from model import ActionSampler

crop_dim = [0, 16, 0, 13]
n_stack = 4
n_skip = 4

# env_wrap = load_smb_env('SuperMarioBros-1-1-v0', crop_dim, n_stack, n_skip)

# states = env_wrap.reset()

# model = PPO.load('models/pre-trained-1', env=env_wrap)

# sample = ActionSampler(model)

# predict = model.predict(states)
# sampled = sample.sample(states)

# print(sampled)
# print(type(predict[0]))


# for noise in [0,0.01,0.2,0.5,1]:
#     new_probs= probs.clone()+noise
#     new_probs = new_probs/new_probs.sum()
#     new_probs:torch.Tensor
#     print(f'noise:{noise} mean:{(new_probs.multinomial(num_samples=100,replacement=True).to(dtype=torch.float32)==3).sum()/100}')


#print(env_wrap.reset_infos[0]['game_screen'].shape)

#states, reward, done, info = env_wrap.step([0])

#print(info[0]['game_screen'].shape)

# from collections import deque

# d = deque()

# d.append(torch.rand(size=(1,5,5)))
# d.append(torch.rand(size=(1,5,5)))
# d.append(torch.rand(size=(1,5,5)))

# d = torch.cat(list(d),dim=0)

# print(d.shape)

d = DDPMSampler(generator=torch.Generator())

x = torch.ones(size=(3,4,5,5))

timesteps = torch.randint(0,500,size=(3,))

print(timesteps)

x,noise = d.add_noise(x,timesteps)

print(x.shape[0])