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


def _zero_padding_tokens(response_tokens):
    mask = response_tokens == PADDING_TOKEN
    assert (
        not (mask[:, :, 1:] < mask[:, :, :-1]).any().item()
    ), f"Padding tokens not a suffix {to_numpy(response_tokens)}"
    return mask, torch.masked_fill(response_tokens, mask, 0)




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
        
    def _eval(queries, responses):
        """
        Run a forward pass. Return all the head values, broadcasted within each replica. If an
        eval_fn is passed, return its output across all replicas.

        :return: A dict with structure:
            eval_stats: structure from eval_fn
            [head]: {
                # disabled for now: query: [batch, query_len+1]
                response: [batch, num_responses, sample_len+1]
            }
        """
        queries = queries.to(self.device)
        responses = responses.to(self.device)
        mask, responses = _zero_padding_tokens(responses)
        responses_per_query = responses.size(1)
        # NOTE: could make this more efficient by sharing context work
        tiled_queries = queries.unsqueeze(1).repeat(1, responses_per_query, 1)
        run_tokens = torch.cat([tiled_queries, responses], dim=2).flatten(0, 1)

        self.model.eval()
        with torch.no_grad():
            outputs = self.head(run_tokens)
        outputs_mb = dict()
            
        reshaped = outputs.view(-1, responses_per_query, outputs.size()[1:])
        d = _split_query_response_output_parts(reshaped, queries.size(1), mask)
        return d

    def forward(self, post_tokens, summary_tokens):
        len_post = post_tokens.shape[-1]
        print(len_post)
        input_ids = torch.concat((post_tokens, summary_tokens), axis=-1)
        print("Shape of input_ids: ", input_ids.shape)

        decoder_input_ids =  torch.concat((post_tokens, summary_tokens), axis=-1)
        print("Shape of decoder_input_ids: ", decoder_input_ids.shape)

        input_ids = input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        outputs = self.supervised_baseline(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # print("Shape of outputs: ", outputs.shape)
        # go through custom layer
        x = outputs.encoder_last_hidden_state
        print("Shape of x: ", x.shape)
        # values = self.head(x.to(self.device))
        # print("Shape of values: ", values.shape)
        # values = values.squeeze(dim=-1)
        # print("Shape of value after squeeze: ", values.shape)
        # 
        # response_values = values[:, len_post:] 
        # response_values = response_values.to(self.device)
        # print("Shape of response_values: ", response_values.shape)

        #new
        response_values = self._eval(post_tokens, summary_tokens)
        
        last_response_indices = _response_indices(summary_tokens)
        print("Shape of last_response_indices: ", last_response_indices.shape)
        last_response_indices = last_response_indices.to(self.device)
        reward = gather_one(
            response_values, last_response_indices, dim=-1
        )
        print("Shape of reward after gather_one: ", reward.shape)
   
        # reward = self.out(y)
        # print("\nREWARD after better_gather: ", reward.shape, "\n", reward)
        # reward = reward.to(self.device)	
        return reward

    def save(self, save_dir, push, repo, key, org):
        self.supervised_baseline.save_pretrained(save_directory=save_dir, push_to_hub=push, repo_url=repo, use_auth_token=key, organization=org)
    def push_to_hub(self, repo):
        self.supervised_baseline.push_to_hub(repo)
