import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

PADDING_TOKEN = -1

def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(
        row_len, dtype=dtype, device=bools.device
    )
    return torch.min(zero_or_index, dim=-1).values

def gather_one(x, indices, *, dim):
    """
    Gather with only one element along the gathered dimension
    """
    return torch.gather(x, dim=dim, index=indices.unsqueeze(dim)).squeeze(dim)

def _response_indices(response_tokens):
    indices = first_true_indices(response_tokens == PADDING_TOKEN) - 1
    return torch.max(indices, torch.zeros([1], dtype=indices.dtype, device=response_tokens.device))


class RewardModel(nn.Module):
    def __init__(self, supervised_baseline, d_model=1024, init_scales=1.0):
        super(RewardModel, self).__init__()

        self.d_model = d_model
        self.supervised_baseline = supervised_baseline
        
        # Add a randomly initialized linear head that outputs a scalar value
        
        init_std = init_scales / np.sqrt(d_model + 1)  #.get(name, 1.0)
        head = nn.Linear(d_model, 1)
        nn.init.normal_(head.weight, std=init_std) #nn.init: initialize weight for a single layer
        nn.init.zeros_(head.bias)
        self.head = head 

    def forward(self, post_tokens, summary_tokens, device=None):
        # print(post_tokens.shape)
        # print(summary_tokens.shape)
        len_post = post_tokens.shape[1] 
        input_ids = torch.concat((post_tokens, summary_tokens), axis=1)
        # print(input_ids.shape)
        decoder_input_ids =  torch.concat((post_tokens, summary_tokens), axis=1)
        # print(decoder_input_ids)
        x = self.supervised_baseline(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # print(x.last_hidden_state.shape)

        # print(input_ids)
        # x = F.pad(input=x, pad=(1, self.d_model - x.shape[1] - 1), mode='constant', value=-1) #value=0 
        # go through custom layer
        
        x = x.last_hidden_state
        if device is not None: 
          values = self.head(x.to(device))
        else: 
          values = self.head(x)
        values = values.squeeze(dim=2)
        # print("\n values.shape: ", values.shape)
        # Call split_ 
        response_values = values[:,len_post:] 
        # print("response_values: ", response_values)
        # print("response_values.shape: ", response_values.shape)
        # call gather_one
        # reward = gather_one(response_values, dim=0, index=torch.LongTensor([[0]]).to(device))#.squeeze(1).squeeze(1)
        # print("REWARD: ", reward)

        last_response_indices = _response_indices(summary_tokens).to(device)
       
        reward = gather_one(
            response_values, last_response_indices, dim=0
        )

        return reward

