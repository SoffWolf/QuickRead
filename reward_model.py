import torch.nn as nn
import numpy as np


class RewardModel(nn.Module):
    def __init__(self, supervised_baseline, d_model=1024, init_scales=0.0):
        super(RewardModel, self).__init__()
        # self.init_scales = init_scales 
        # self.d_model = d_model
        self.supervised_baseline =  supervised_baseline

        # Add a randomly initialized linear head that outputs a scalar value
        init_std = init_scales / np.sqrt(d_model + 1)  #.get(name, 1.0)
        head = nn.Linear(d_model, 1)
        nn.init.normal_(head.weight, std=init_std)
        nn.init.zeros_(head.bias)
        self.head = head

    def forward(self, post_tokens, summary_tokens):
        x = self.supervised_baseline(post)
        y = self.NeuralNet(summary)
        z = torch.stack(x, y)
        z = self.n2(z)
        # go through custom layer
        reward = self.head(z)
        return reward



