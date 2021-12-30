import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, supervised_baseline, d_model=64, init_scales=1.0):
        super(RewardModel, self).__init__()
        self.d_model = d_model
        self.supervised_baseline = supervised_baseline
        # Add a randomly initialized linear head that outputs a scalar value

        init_std = init_scales / np.sqrt(d_model + 1)  #.get(name, 1.0)
        head = nn.Linear(d_model, 1)
        nn.init.normal_(head.weight, std=init_std)
        nn.init.zeros_(head.bias)
        self.head = head

    def forward(self, post_tokens, summary_tokens):
        x = torch.concat((post_tokens, summary_tokens), axis=1) 
        x = self.supervised_baseline.generate(input_ids=x)
        x = F.pad(input=x, pad=(1, self.d_model - x.shape[1] - 1), mode='constant', value=0)
        print(x.shape)
        # go through custom layer
        reward = self.head(x.type(torch.FloatTensor))
        return reward



