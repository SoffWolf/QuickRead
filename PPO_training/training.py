import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
import sys
sys.path.insert(0,'..')

from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from pegasus_with_heads import PegasusWithValueHead
from ppo import PPOTrainer
from rewards.reward_model import RewardModel


config = {
    "lm_name": "lvwerra/gpt2-imdb",   # policy: supervised baseline
    "ref_lm_name": "lvwerra/gpt2-imdb",   # find out about the ref model
    "cls_model_name": "lvwerra/distilbert-imdb",   # reward model
    "tk_name": "gpt2",    # tokenizer name
    "steps": 25600,
    "batch_size": 256,
    "forward_batch_size": 16,
    "ppo_epochs": 5,   
    "txt_in_len": 5,
    "txt_out_len": 15,
    "lr": 1.41e-5,    # check this in the paper
    "init_kl_coef":0.2,   # check this in the paper
    "target": 6,
    "horizon":10000,
    "gamma":1,    # also check these in the paper
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

## WANDB 
group = "quickread"
project = "PPO-training"
display_name = "experiment-2022-27-1"
wandb.init(entity=group, project=project, name=display_name, config=config)



# load supervised baseline
supervised_baseline = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large", cache_dir="HF_HOME")

# Reward model
reward_model = RewardModel(supervised_baseline)

# Policy model
#policy = PegasusWithValueHead(supervised_baseline)
#policy_ref = PegasusWithValueHead(supervised_baseline)
policy = supervised_baseline
policy_ref = supervised_baseline

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large", cache_dir="HF_HOME")

# Wandb
wandb.watch(policy, log='all')

# Put all the model to cuda, if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = supervised_baseline.to(device)
_ = reward_model.to(device)
_ = policy.to(device)
_ = policy_ref.to(device)

# Load the data 
dataset = load_from_disk("../../../QuickRead/reddit_clean")
train_texts, train_labels = dataset['train']['content'], dataset['train']['summary']
val_texts, val_labels = dataset['valid']['content'], dataset['valid']['summary']
test_texts, test_labels = dataset['test']['content'], dataset['test']['summary']

# Tokenize the data
#def tokenize(inp):
#    df = pd.DataFrame(inp)
#    print(type(df[0].to_string()))
#    print(df[0].to_string())
#    ret = df.progress_apply(lambda x: tokenizer(x.to_string(), return_tensors="pt").to(device))
#    print("ret: ", ret)
#    return ret

#token_train, token_train_sum = tokenize(train_texts), tokenize(train_labels)
#token_val, token_val_sum = tokenize(val_texts), tokenize(val_labels)
#token_test, token_test_sum = tokenize(test_texts), tokenize(test_labels)

#################### Training ######################
ppo_trainer = PPOTrainer(policy, policy_ref, **config)
fbs = config['forward_batch_size']
epochs = 2
for epoch in range(epochs):
    logs = dict()
    timing = dict()
    t0 = time.time()

    query_tensors = []  # get query tensor for PPO training
    response_tensors = []
    rewards = []
    
    for i in range (2):
        #print(type(train_texts), type(train_texts[0]))
        query = tokenizer(train_texts[i]).input_ids
        #print(query)
    #for query, label in tqdm(token_train[:1000]):
        query = torch.LongTensor(query).unsqueeze(0)
        # print("query: ", query.shape)
        #logits, response, values = policy(query)
        response = policy.generate(input_ids=query)
        # print("response: ", response.shape)
        response = torch.LongTensor(response)
        reward = reward_model(query, response)
        query_tensors.append(query.squeeze(0))
        response_tensors.append(response.squeeze(0))
        rewards.append(reward)

    #print("query_tensors: ", query_tensors.shape, query_tensors[0].shape,query_tensors[1].shape )
    #print("response_tensors: ", response_tensors.shape, response_tensors[0].shape,response_tensors[1].shape )
    print("rewards: ", rewards)
    query_tensors = torch.nn.utils.rnn.pad_sequence(query_tensors)
    response_tensors = torch.nn.utils.rnn.pad_sequence(response_tensors)
    print("padded query tensor: ", query_tensors.shape)
    print("padded response_tensors: ", response_tensors)   
    query_tensors = query_tensors.unsqueeze(dim=0)
    response_tensors = response_tensors.unsqueeze(dim=0)
    rewards = torch.cat(rewards)
    print("padded query tensor: ", query_tensors.shape)
    print("padded response_tensors: ", response_tensors)   
        
    #### Run PPO training 
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
     
    #### Log everything
    timing['time/epoch'] = time.time()-t0
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    wandb.log(logs)
# Save model
os.makedirs('result')
policy.save_pretrained('ppo_fine_tune')
tokenizer.save_pretrained('ppo_fine_tune_tokenizer')
