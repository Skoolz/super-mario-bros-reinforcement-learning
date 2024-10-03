import gym

class GameObsInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['game_screen'] = obs
        return obs, reward, terminated, truncated, info
    
    def reset(self,seed=None,options=None):
        obs,info = self.env.reset(seed,options)

        info['game_screen'] = obs
        return obs,info