from stable_baselines3.common.policies import obs_as_tensor

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