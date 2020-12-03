import gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Flatten()
        )
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.SELU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))
