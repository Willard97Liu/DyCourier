import torch.nn as nn
from copy import deepcopy

import torch

class NN(nn.Module):#(BaseFeaturesExtractor):

    def __init__(self, 
                #  observation_space: spaces.Box, 
                 n_observation, 
                 hidden_layers = [512, 512, 256],
                 n_actions: int = 1):
        super().__init__()
        hidden = deepcopy(hidden_layers)
        hidden.insert(0, n_observation)
        layers = []
        for l in range(len(hidden)-1):
            layers += [
                nn.Linear(hidden[l], hidden[l+1]),
                nn.ReLU()
            ]
        layers += [
            nn.Linear(hidden[-1], n_actions),
            # nn.Sigmoid()
            # nn.Softmax()
        ]

        self.linear = nn.Sequential(
            *layers
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)
    
    
