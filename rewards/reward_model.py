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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.head = self.head.to(self.device)
        
    def forward(self, post_tokens, summary_tokens):
        len_post = post_tokens.shape[-1]
        input_ids = torch.concat((post_tokens, summary_tokens), axis=-1)
        decoder_input_ids =  torch.concat((post_tokens, summary_tokens), axis=-1)
        input_ids = input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        outputs = self.supervised_baseline(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # print("Shape of outputs: ", outputs.shape)
        # go through custom layer
        x = outputs.encoder_last_hidden_state
        values = self.head(x.to(self.device))
        values = values.squeeze(dim=-1)
        
        response_values = values[:, len_post:] 
        response_values = response_values.to(self.device)

        last_response_indices = summary_tokens.shape[1] - 1
        last_response_indices = torch.tensor([last_response_indices])
        last_response_indices = last_response_indices.to(self.device)
        reward = gather_one(
            response_values, last_response_indices, dim=-1
        )
   
        return reward

    def save(self, save_dir, push, org):
        self.supervised_baseline.save_pretrained(save_directory=save_dir, push_to_hub=push, organization=org)

    def push_to_hub(self, repo):
        self.supervised_baseline.push_to_hub(repo)
