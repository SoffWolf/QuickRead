import torch
import torch.nn as nn
import numpy as np

def build_model(model_type, vec_length, learning_rate=None):
    if 'linear' in model_type:
        deep_model = torch.nn.Sequential(
            torch.nn.Linear(vec_length, 1),
        )
    else:
        deep_model = torch.nn.Sequential(
            torch.nn.Linear(vec_length, int(vec_length/2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(vec_length/2), 1),
        )
    if learning_rate is not None:
        optimiser = torch.optim.Adam(deep_model.parameters(),lr=learning_rate)
        return deep_model, optimiser
    else:
        return deep_model

class RewardModel(nn.Module):
    def __init__(self, supervised_baseline, d_model=1024, init_scales=0.0):
        super(RewardModel, self).__init__()
        # self.init_scales = init_scales 
        # self.d_model = d_model
        self.supervised_baseline = supervised_baseline

        # Add a randomly initialized linear head that outputs a scalar value
        init_std = init_scales / np.sqrt(d_model + 1)  #.get(name, 1.0)
        head = nn.Linear(d_model, 1)
        nn.init.normal_(head.weight, std=init_std)
        nn.init.zeros_(head.bias)
        self.head = head

        self.reward_model=build_model("Pegasus", d_model)
    def forward(self, post_tokens, summary_tokens):
        x = self.supervised_baseline(post)
        y = self.NeuralNet(summary)
        z = torch.stack(x, y)
        z = self.n2(z)
        # go through custom layer
        reward = self.head(z)
        return reward



