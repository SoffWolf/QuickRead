# value head init from reward model weight
import numpy as np
import torch.nn.functional as F
import torch

from transformers import top_k_top_p_filtering, PegasusPreTrainedModel
from torch import nn
from torch.nn import Identity
from rewards.reward_model import RewardModel


class ValueHead(nn.Module):
    """The ValueHead class implements a head for Pegasus that returns a scalar for each output token."""
    def __init__(self, d_model=1024, init_scales=1.0):
        super().__init__()
        self.detach_head = False
        init_std = init_scales / np.sqrt(d_model + 1)  #.get(name, 1.0)
        head = nn.Linear(d_model, 1)
        nn.init.normal_(head.weight, std=init_std) #nn.init: initialize weight for a single layer
        nn.init.zeros_(head.bias)
        self.head = head

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        value = self.head(output)
        return value


class PegasusWithValueHead(nn.Module):
    """This model class is Pegasus language model with a secondary scalar head"""
    def __init__(self, supervised_baseline, d_model=1024, vocab_size = 96103, init_scales=1.0):
        super(PegasusWithValueHead,self).__init__()
        self.d_model = d_model
        self.model = supervised_baseline
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.v_head = ValueHead()

    def get_output_embeddings(self):
        return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True


    def forward(self, post_tokens, device=None):
        # len_post = post_tokens.shape[1]
        #print("Pegasus with head forward: ", post_tokens.shape, type(post_tokens))  
        input_ids = post_tokens
        decoder_input_ids = input_ids
        x = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        #print("x: ", x)
        #print("x.encoder_last_hidden_state: ", x.encoder_last_hidden_state)
        # go through custom layer
        hidden_states = x.encoder_last_hidden_state
        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)
        print("lm_logits.shape : ", lm_logits.shape) 
        print("Value after squeeze: ", value.shape)
        outputs = (lm_logits,) + (torch.zeros((1,1)),) + (value,)
        return outputs
    def generate(self, post_tokens):
        input_ids = post_tokens
        output_token = self.model.generate(input_ids = input_ids)
        return output_token

    def save(self, save_dir, push, repo, key, org):
        self.model.save_pretrained(save_directory=save_dir, push_to_hub=push, repo_url=repo, use_auth_token=key, organization=org)
    def push_to_hub(self, repo):
        self.model.push_to_hub(repo)


        # if device is not None: 
        #   values = self.head(x.to(device))
        # else: 
        #   values = self.head(x)
        # values = values.squeeze(dim=2)
        # # Call split_ 
        # response_values = values[:,len_post:] 
        # response_values = response_values.to(device)

        # last_response_indices = _response_indices(summary_tokens)
        # last_response_indices = last_response_indices.to(device)
        # reward = gather_one(
        #     response_values, last_response_indices, dim=0
        # )

        # return reward


def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]
