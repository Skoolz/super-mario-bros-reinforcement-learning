from gym_utils import *
import torch

from pstd.sd.ddpm import DDPMSampler

crop_dim = [0, 16, 0, 13]
n_stack = 4
n_skip = 4

#env_wrap = load_smb_env('SuperMarioBros-1-1-v0', crop_dim, n_stack, n_skip)

#states = env_wrap.reset()

#print(env_wrap.reset_infos[0]['game_screen'].shape)

#states, reward, done, info = env_wrap.step([0])

#print(info[0]['game_screen'].shape)

d = DDPMSampler(generator=torch.Generator())

x = torch.ones(size=(2,4,5,5))

x,noise = d.add_noise(x,torch.tensor([999,0],dtype=torch.int32))

print(noise.shape)