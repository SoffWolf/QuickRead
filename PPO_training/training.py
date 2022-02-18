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
    "lm_name": "SophieTr/fine-tune-Pegasus",   # policy: supervised baseline
    "ref_lm_name": "SophieTr/fine-tune-Pegasus",   # find out about the ref model
    "cls_model_name": "SophieTr/fine-tune-Pegasus",   # reward model
    "tk_name": "gpt2",    # tokenizer name
    "steps": 25600,
    "batch_size": 8,
    "forward_batch_size":4,
    "ppo_epochs": 1,   
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
supervised_baseline = PegasusForConditionalGeneration.from_pretrained("SophieTr/fine-tune-Pegasus", cache_dir="HF_HOME")

# Reward model
reward_model = RewardModel(supervised_baseline)
reward_model.load_state_dict(torch.load(os.path.join("../rewards/reward_model_weight/epoch-1.pth")), strict=False)

# Policy model
policy = PegasusWithValueHead(supervised_baseline)
policy_ref = PegasusWithValueHead(supervised_baseline)
#policy = supervised_baseline
#policy_ref = supervised_baseline

keys_file = open("hfAPI.txt")
key = keys_file.readlines()[0].rstrip()
#print(key)

tokenizer = PegasusTokenizer.from_pretrained("SophieTr/fine-tune-Pegasus", cache_dir="HF_HOME")

save_directory = "QuickRead/PPO_training"
#policy.save(save_directory, True, 'https://huggingface.co/QuickRead/PPO_training', key, "QuickRead")
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

df = pd.DataFrame(train_texts)
#print("DF: ",(df))
 
#################### Training ######################
ppo_trainer = PPOTrainer(policy, policy_ref, **config)
fbs = config['forward_batch_size']
#train_texts, val_texts, test_texts = train_texts.to(device), val_texts.to(device), test_texts.to(device)

for epoch in tqdm(range(int(np.ceil(len(train_texts) / config["batch_size"])))):
    torch.cuda.empty_cache()
    logs = dict()
    timing = dict()
    t0 = time.time()

    query_batch = df.sample(config["batch_size"])

    query_tensors = []  # get query tensor for PPO training
    response_tensors = []
    rewards = []
    
    for i in range(int(config["batch_size"] / fbs)):
        query = query_batch[i*fbs:(i+1)*fbs]
        query = map(lambda x: x[0], query.values.tolist())
        #print(type(query), query.items())
        query = list(query)
        query = tokenizer(query, padding=True, truncation=True, return_tensors='pt').input_ids
        #print("QUERY after tokenizer: ", query, type(query))
        query = query.to(device)
        #print("query: ", query.shape)
        #logits, response, values = policy(query)
        response = policy.generate(query)
        #response = torch.FloatTensor(response)
        #response = response.to(device)
        try:
            reward = reward_model(query, response).detach()
        except:
            pass
        #for k in range(fbs):
        #    query_tensors = query_tensors.append(query[k])
        #    response_tensors = response_tensors.append(response[k])
        query_tensors = query_tensors + list(torch.split(query,1))
        response_tensors = response_tensors + list(torch.split(response,1))
        rewards.append(reward)
    for k in range(len(query_tensors)):
        query_tensors[k] = query_tensors[k].squeeze(0)
        response_tensors[k] = response_tensors[k].squeeze(0)
    #print("query_tensors: ", len(query_tensors), query_tensors[0].shape,query_tensors[1].shape )
    #print("response_tensors: ", len(response_tensors), response_tensors[0].shape,response_tensors[1].shape )
    query_tensors = torch.nn.utils.rnn.pad_sequence(query_tensors)
    response_tensors = torch.nn.utils.rnn.pad_sequence(response_tensors)
    query_tensors = query_tensors.unsqueeze(dim=0).to(device)
    response_tensors = response_tensors.unsqueeze(dim=0).to(device)
    rewards = torch.cat(rewards).to(device)
    
    query_tensors = query_tensors.view(query_tensors.shape[2], query_tensors.shape[1])
    response_tensors = response_tensors.view(response_tensors.shape[2], response_tensors.shape[1])
    #print("query_tensors: ", query_tensors.shape)
    #print("response_tensors: ", response_tensors.shape)
    #print("rewards: ", rewards.shape)

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
    
    ## Push model to hub every 6000 epoch
    if (epoch+1) % 4000 == 0:
        print("EPOCH: ", epoch)
        # HF push_to_hub:
        policy.push_to_hub("QuickRead/PPO_training")
        tokenizer.push_to_hub("QuickRead/PPO_training")


# Save model
checkpoint = {'state_dict': policy.state_dict()}
#torch.save(checkpoint, os.path.join("./result/test.pth"))
torch.save(checkpoint, os.path.join("./ppo_checkpoints_newRM", 'epoch-{}.pth'.format(epoch+1)))
   
