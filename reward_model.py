import torch.nn as nn
import numpy as np


class RewardModel(nn.Module):
    def __init__(self, supervised_baseline, d_model=1024, init_scales=0.0):
        super(RewardModel, self).__init__()

        self.post_input = supervised_baseline
        self.summary_input = supervised_baseline
        self.final = nn.Sequential(nn.Linear(2*d_model, d_model))

        # Add a randomly initialized linear head that outputs a scalar value
        init_std = init_scales / np.sqrt(d_model + 1)  #.get(name, 1.0)
        head = nn.Linear(d_model, 1)
        nn.init.normal_(head.weight, std=init_std)
        nn.init.zeros_(head.bias)
        self.head = head

    def forward(self, post_tokens, summary_tokens):
        x = self.post_input(post)
        y = self.summary_input(summary)
        z = torch.stack([x, y], 1)
        z = self.final(z)
        # go through custom layer
        reward = self.head(z)
        return reward



