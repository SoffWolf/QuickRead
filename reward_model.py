import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, supervised_baseline):
        super(RewardModel, self).__init__()
        self.supervised_baseline= supervised_baseline
        self.reward_head = nn.Sequential(
                nn.Linear(..., ...)) #check the final layer of Pegasus and decide how to write the " randomly initialized linear head that outputs a scalar value" here

    def forward(self, inputs):
        x = self.model(inputs)
        # go through custom layer
        x = self.reward_head(x)
        return x
