import torch
import torch.nn as nn
import torch.functional as F
from reward_model import RewardModel


supervised_baseline = load_the_model("SophieTr/supervised_baseline")
model = RewardModel(supervised_baseline, d_model=final_layer_size)

#### Write the training loop here
