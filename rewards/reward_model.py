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
    print("Indices.shape = ", indices.shape, "\nIndices:\n\t", indices, "\nindices.unsqueeze(", dim, "): \n\t", indices.unsqueeze(dim))
    return torch.gather(x, dim=dim, index=indices.unsqueeze(dim)).squeeze(dim)

def _response_indices(response_tokens):
    indices = first_true_indices(response_tokens == PADDING_TOKEN) - 1
    print("\nResult from first true indices:\n\t ", indices)
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
        #print("Before device 1")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        #print("after self.device")
        self.head = self.head.to(self.device)
        self.fix_length = 512
        
        # New NN for better gather_one
        self.lin = nn.Linear(self.fix_length, 10) 
        self.dropout = nn.Dropout(0.1)                # dropout layer
        self.out = nn.Linear(10, len(self.word2idx)+1) # output layer
        #_ = self.better_gather.to(self.device)
        #print("After self.better_gather.to(device)")

    def forward(self, post_tokens, summary_tokens):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print("Device = ", device)
        len_post = post_tokens.shape[-1]
        #print("After len_post")
        input_ids = torch.concat((post_tokens, summary_tokens), axis=-1)
        #print("After input_ids")

        decoder_input_ids =  torch.concat((post_tokens, summary_tokens), axis=-1)
        #print("After decoder_input_ids")

        input_ids = input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        #print("After input_ids, decoder_ids . to(Device)")
        outputs = self.supervised_baseline(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        print("After output")
        # go through custom layer
        x = outputs.encoder_last_hidden_state
        print("After x")
        #if device is not None: 
        values = self.head(x.to(self.device))
        #else: 
          #values = self.head(x)
        values = values.squeeze(dim=-1)
        print("Values.shape: ", values.shape)
        reward = F.pad(input=values, pad = (1, self.fix_length - values.shape[1] - 1), mode='constant', value=0) 
        # Call split_ 
        #response_values = values[:, len_post:] 
        #response_values = response_values.to(device)
        #print("response_values: ", response_values)
        # call gather_one
        # reward = gather_one(response_values, dim=0, index=torch.LongTensor([[0]]).to(device))#.squeeze(1).squeeze(1)
        
        #last_response_indices = _response_indices(summary_tokens)
        #last_response_indices = torch.tensor([0])
        #last_response_indices = last_response_indices.to(device)
        #reward = gather_one(
        #    response_values, last_response_indices, dim=-1
        #)
   
        print("\nREWARD before better_gather: ", reward.shape, "\n", reward)
        y = self.lin(reward)
        y = F.relu(y)
        y = self.dropout(y)
        
        reward = self.out(y)
        print("\nREWARD after better_gather: ", reward.shape, "\n", reward)
        reward = reward.to(self.device)	
        return reward

    def save(self, save_dir, push, repo, key, org):
        self.supervised_baseline.save_pretrained(save_directory=save_dir, push_to_hub=push, repo_url=repo, use_auth_token=key, organization=org)
    def push_to_hub(self, repo):
        self.supervised_baseline.push_to_hub(repo)
