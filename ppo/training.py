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
from ../rewards/reward_model.py import RewardModel


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
supervised_baseline = PegasusForConditionalGeneration.from_pretrained("SophieTr/fine-tune-Pegasus-large")

# Reward model
reward_model = RewardModel(supervised_baseline)

# Policy model
policy = PegasusWithValueHead(supervised_baseline)
policy_ref = PegasusWithValueHead(supervised_baseline)
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large", cache_dir="HF_HOME")

# Put all the model to cuda, if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = supervised_baseline.to(device)
_ = reward_model.to(device)
_ = policy.to(device)
_ = policy_ref.to(device)
_ = tokenizer.to(device)

# Load the data 
dataset = load_from_disk("../QuickReadOld/reddit_clean")
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

for epoch in range(epochs):
    logs = dict()
    timing = dict()
    t0 = time.time()

    query_tensors = []  # get query tensor for PPO training
    response_tensors = []
    rewards = []
    for query, label in tqdm(token_train[:1000]):
        logits, response, values = policy(query)
        reward = reward_model(query, response)
        query_tensors.append(query)
        response_tensors.append(response)
        rewards.append(reward)

    query_tensors = torch.cat(query_tensors)
    response_tensors = torch.cat(response_tensors)
    rewards = torch.cat(rewards)
    
    #### Run PPO training 
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
     
    #### Log everything

# Save model
os.makedirs('result')
policy.save_pretrained('ppo_fine_tune')
tokenizer.save_pretrained('ppo_fine_tune_tokenizer')
