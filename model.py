from stable_baselines3.common.policies import obs_as_tensor
import numpy as np

class ActionSampler:
    def __init__(self,model,noise_level=0) -> None:
        self.model = model
        self.set_noise_level(noise_level)

    def set_noise_level(self,noise_level):
        self.noise_level = noise_level
    
    def sample(self,obs):
        obs = obs_as_tensor(obs,device='cpu')

        dist = self.model.policy.get_distribution(obs)

        probs = dist.distribution.probs.detach()

        probs += self.noise_level

        probs = probs/probs.sum()

        action = probs.multinomial(num_samples=1)

        return action.view(-1).numpy()

class EpsilonGreedyActionSampler:
    def __init__(self, model, epsilon=-1):
        self.model = model

        if(epsilon == -1):
            self.epsilon = lambda : np.random.uniform(0,0.1)
        else:
            self.epsilon = lambda : epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def sample(self, obs):
        if np.random.rand() < self.epsilon():
            # Выбираем случайное действие с вероятностью ε
            action = np.random.randint(0,self.model.action_space.n,size=(1,))
        else:
            # Иначе выбираем действие от модели
            obs = obs_as_tensor(obs, device='cpu')
            dist = self.model.policy.get_distribution(obs)
            action = dist.get_actions().cpu().numpy()
        return action
