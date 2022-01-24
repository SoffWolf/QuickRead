import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt

from pegasus_with_heads import PegasusWithValueHead
from ppo import PPOTrainer
from ../rewards.py import RewardModel


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



# load supervised baseline
supervised_baseline = AutoModelForSeq2SeqLM.from_pretrained("SophieTr/results")

# Reward model
reward_model = RewardModel(supervised_baseline)

# Policy model
policy = PegasusWithValueHead(supervised_baseline)
policy_ref = PegasusWithValueHead(supervised_baseline)
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")

# Put all the model to cuda, if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = supervised_baseline.to(device)
_ = reward_model.to(device)
_ = policy.to(device)
_ = policy_ref.to(device)
_ = tokenizer.to(device)

# Load the data 
dataset = load_from_disk("reddit_clean")
train_texts, train_labels = dataset['train']['content'], dataset['train']['summary']
val_texts, val_labels = dataset['valid']['content'], dataset['valid']['summary']
test_texts, test_labels = dataset['test']['content'], dataset['test']['summary']

# Tokenize the data
def tokenzize(df):
    ret = df.progress_apply(lambda x: tokenizer(x, return_tensors="pt").to(device))
    return ret

token_train, token_train_sum = tokenize(train_texts), tokenize(train_labels)
token_val, token_val_sum = tokenize(val_texts), tokenize(val_labels)
token_test, token_test_sum = tokenize(test_texts), tokenize(test_labels)

#################### Training ######################
ppo_trainer = PPOTrainer(policy, policy_ref, **config)
fbs = config['forward_batch_size']

for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
    torch.cuda.empty_cache()
    logs = dict()
    game_data = dict()
    timing = dict()
    t0 = time.time()
    
    #### get a batch from the dataset
    query = train_texts.sample(config['batch_size'])
    query_token = token_train.sample(config['batch_size'])
    game_data['query'] = query.tolist()
    query_tensors = torch.stack(query_token.tolist())
    
    #### get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(int(config['batch_size']/fbs)):
        response  = respond_to_batch(policy, query_tensors[i*fbs:(i+1)*fbs],
                                     txt_len=config['txt_out_len'])
        response_tensors.append(response)
    response_tensors = torch.cat(response_tensors)
    game_data['response'] = [tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]

    #### tokenize text for sentiment analysis
    t = time.time()
    texts = [q + r for q,r in zip(game_data['query'], game_data['response'])]

    #### get sentiment score
    rewards = []
    for i in range(int(config['batch_size']/fbs)):
        res = reward_model.forward(game_data['query'][i*fbs:(i+1)*fbs],
                                    game_data['response'][i*fbs:(i+1)*fbs])
        rewards.append(res)
    rewards = torch.cat(rewards)
    
    #### Run PPO training 
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
     
    #### Log everything
    table_rows = [list(r) for r in zip(game_data['query'], game_data['response'], rewards.cpu().tolist())]
    logs.update({'game_log':wandb.Table(
        columns=['query', 'response', 'reward'],
        rows=table_rows)})
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
