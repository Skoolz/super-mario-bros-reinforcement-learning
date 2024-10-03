from gym_utils import *
import torch

crop_dim = [0, 16, 0, 13]
n_stack = 4
n_skip = 4

#env_wrap = load_smb_env('SuperMarioBros-1-1-v0', crop_dim, n_stack, n_skip)

#states = env_wrap.reset()

#print(env_wrap.reset_infos[0]['game_screen'].shape)

#states, reward, done, info = env_wrap.step([0])

#print(info[0]['game_screen'].shape)

loss = (torch.rand(size=(10,5,5,3))-torch.rand(size=(10,5,5,3)))**2

loss = loss.mean()

print(loss.shape)