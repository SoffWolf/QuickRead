import torch
# import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
import sys
sys.path.insert(0,'..')
from sklearn.utils import shuffle

from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from pegasus_with_heads import PegasusWithValueHead
from ppo import PPOTrainer
from rewards.reward_model import RewardModel

# from torchsummary import summary

config = {
    "lm_name": "QuickRead/pegasus-reddit-7e05",   # policy: supervised baseline
    "ref_lm_name": "QuickRead/pegasus-reddit-7e05",   # find out about the ref model
    "cls_model_name": "SophieTr/RM_incr_lr_v1",   # reward model
    "tk_name": "QuickRead/pegasus-reddit-7e05",    # tokenizer name
    "steps": 25600,
    "batch_size": 8, #TO BE BACK TO 8
    "forward_batch_size":1,
    "ppo_epochs": 1,   
    "txt_in_len": 5,
    "txt_out_len": 15,
    "lr": 1.41e-5,          # check this in the paper
    "init_kl_coef":0.2,     # check this in the paper
    "target": 6,
    "horizon":10000,
    "gamma":1,              # also check these in the paper
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

RUN_NAME = "PPO_print_fail"
RM_name = "RM_incr_lr_v4_no_wandb" #"RM_incr_lr_v1"
RM_PATH = "../rewards/" + RM_name +  "/epoch-1.pth"
PATH = "./" + RUN_NAME
CHECKPOINT_PATH = os.path.join(PATH, 'latest_minibatch.pth')

## WANDB 
group = "quickread"
project = "PPO-training"
display_name = RUN_NAME
# wandb.init(entity=group, project=project, name=display_name, config=config)

# load supervised baseline
supervised_baseline = PegasusForConditionalGeneration.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")

# Reward model
reward_model = RewardModel(supervised_baseline)
reward_model.load_state_dict(torch.load(os.path.join(RM_PATH)), strict=False)
#reward_model.load_state_dict(torch.load(os.path.join(RM_PATH),map_location=torch.device('cpu')), strict=False)

# Policy model
policy = PegasusWithValueHead(supervised_baseline)
policy_ref = PegasusWithValueHead(supervised_baseline)
tokenizer = PegasusTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")

# save_directory = RUN_NAME
# policy.save(save_directory, True, "QuickRead")
# Wandb
# wandb.watch(policy, log='all')

# Put all the model to cuda, if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

_ = supervised_baseline.to(device)
_ = reward_model.to(device)
_ = policy.to(device)
_ = policy_ref.to(device)

# Load the data 
dataset = load_from_disk("../../../QuickRead/reddit_clean")
train_texts, train_labels = dataset['train']['content'], dataset['train']['summary']
val_texts, val_labels = dataset['valid']['content'], dataset['valid']['summary']
test_texts, test_labels = dataset['test']['content'], dataset['test']['summary']

df = pd.DataFrame(train_texts) # Add a Dataset & Dataloader class to handle loading data from disk, & shuffle & sample
# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
print(pd.DataFrame(train_texts).head(n=5), pd.DataFrame(train_labels).head(n=5))

#################### Training ######################
ppo_trainer = PPOTrainer(policy, policy_ref, **config)
fbs = config['forward_batch_size']
#train_texts, val_texts, test_texts = train_texts.to(device), val_texts.to(device), test_texts.to(device)

if not os.path.exists(PATH):
    print('The path to latest checkpoint NOT exist')
    os.mkdir(RUN_NAME)
else:
    # load check points 
    print("Resumed training from last saved epoch")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    policy.load_state_dict(checkpoint['state_dict'])
    policy.to(device)
    ## Now the epoch is still gonna be retrained

error_lst = []
for epoch in range(1):
    # sample = shuffle(df)
    sample = df
    if len(sample) != df.shape[0]:
        print("IN BREAK", flush=True)
        break
    for z in range(8):
        print(sample[22016+z])
    break
    for k in range(0, int(np.ceil(len(sample)))-config["batch_size"], config["batch_size"]): #tqdm(range(int(np.ceil(len(sample) / config["batch_size"])))):
    #for k in range( int(np.ceil(len(sample)/2)), int(np.ceil(len(sample))-config["batch_size"]) ): #, config["batch_size"]):
        # print("k: ", k, flush=True)
            query_batch = sample[k:k+config["batch_size"]]
            logs = dict()
            timing = dict()
            t0 = time.time()

            query_tensors = []  # get query tensor for PPO training
            response_tensors = []
            rewards = []
            
            for i in range(int(config["batch_size"] / fbs)):
                try:
                    query = query_batch[i*fbs:(i+1)*fbs]
                    query = map(lambda x: x[0], query.values.tolist())
                    query = list(query)
                    query = tokenizer(query, padding=True, truncation=True, return_tensors='pt').input_ids
                    query = query.to(device)
                except Exception as e1:
                    print('_*_'*100)
                    print('Possible indexing error at catch e1 for tokenizing query: ', e1)
                    print('Tokenizer config: ', tokenizer)
                    print('Length of query ib question: ', len(query))
                    break

                # print("QUERY (", i, ") = ",query.shape)
                try:
                    response = policy.generate(query) # will not produce text
                    response = response.to(device)
                except Exception as e2:
                    print('_*_'*100)
                    print('Possible indexing error at catch e2 for generating response using policy: ', e2)
                    print('Policy config: ', policy.model)
                    print('Length of query in question: ', len(query))
                    break   

                if not (torch.all(query >= 0)).item():
                    print('query in text form: \n', query_batch[i*fbs:(i+1)*fbs])
                    print('query as input_ids: \n', query)
                    print('corresponding response to FAIL query as input_ids: \n', response)
                if not (torch.all(response >= 0)).item():
                    print('corresponding query to FAIL response in text form: \n', query_batch[i*fbs:(i+1)*fbs])
                    print('corresponding query to FAIL response as input_ids: \n', query)
                    print('response as input_ids: \n', response)

                if i == 0:
                    if k == 0 or k%500 == 0:
                        resp_txt = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        print(f'RESPONSE txt("{i}") from mini batch ("{k}") is:\n {resp_txt}', flush=True)
                
                try:
                    reward_model.eval()
                    with torch.no_grad():
                        reward = reward_model(query, response).detach()
                    reward = reward.to(device)
                except Exception as e3:
                    print('_*_'*100)
                    print('Possible indexing error at catch e3 for generating rewards with reward_model: ', e3)
                    print('Reward model config: ', reward_model)
                    print('Length of query and response in question: ', query.size(), response.size())
                    print(f'k = {k}, i = {i}')
                    print('FAIL query: ', query)
                    print('FAIL response: ', response)
                    query_tobe_blacklisted = query_batch[i*fbs:(i+1)*fbs]
                    queries_array_contained_black_listed_query = query_batch
                    with open("BLACK_LISTED.txt","a") as file:
                        file.write("\n","__*__"*100, "\n")
                        file.write("Query array containing the black listed query: \n")
                        for i in queries_array_contained_black_listed_query:
                            file.write('__', i, '\n')
                        file.write('-->.<--' * 100, '\n')
                        file.write("query_tobe_blacklisted: \n\n")
                        file.write(query_tobe_blacklisted, '\n')
                        file.write('-->.<--' * 100, '\n')
                    break
                query_tensors = query_tensors + list(torch.split(query,1))

                response_tensors = response_tensors + list(torch.split(response,1))

                rewards.append(reward)
            for j in range(len(query_tensors)):
                query_tensors[j] = query_tensors[j].squeeze(0)
                response_tensors[j] = response_tensors[j].squeeze(0)
            try:
                query_tensors = torch.nn.utils.rnn.pad_sequence(query_tensors)
                response_tensors = torch.nn.utils.rnn.pad_sequence(response_tensors)
                query_tensors = query_tensors.unsqueeze(dim=0).to(device)
                response_tensors = response_tensors.unsqueeze(dim=0).to(device)
                # print("Rewards before torch.cat: ", rewards)
                rewards = torch.cat(rewards).to(device)
                # print("Rewards after torch.cat: ", rewards)
                query_tensors = query_tensors.view(query_tensors.shape[2], query_tensors.shape[1])
                response_tensors = response_tensors.view(response_tensors.shape[2], response_tensors.shape[1])

                #### Run PPO training --> COMMENT THIS OUT FOR NEXT TEST TO COLECT error data points
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            except Exception as e4:
                print('_*_'*100)
                print('Possible indexing error at catch e4 for geting stats with ppo_trainer: ', e4)
                print('ppo_trainer config: ', ppo_trainer)
                print('Shape of query tensors, response tensors, and rewards in question: ', query_tensors.shape, response_tensors.shape, len(rewards))
                break
            #### Log everything
            timing['time/mini_batch'] = time.time()-t0
    #         logs.update(timing)
    #         logs.update(stats)
    #         logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    #         logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    #         logs['env/reward_dist'] = rewards.cpu().numpy()
            # wandb.log(logs)
            if k>1 and k%1000 == 0:
                #TODO: print model output. Verify same model is being save
                print("At k = ", k, " ---> Reward mean = ", torch.mean(rewards).cpu().numpy(), flush=True)
                checkpoint = {'state_dict': policy.state_dict(), 'mini_batch': k}
                torch.save( checkpoint, os.path.join(PATH, 'latest_minibatch.pth') )

# HF push_to_hub:
# policy.push_to_hub("SophieTr/"+RUN_NAME)
# tokenizer.push_to_hub("SophieTr/"+RUN_NAME)
file=open('error_2.txt','w')
for items in error_lst:
    file.writelines(items+'\n')
file.close()
checkpoint = {'state_dict': policy.state_dict()}
torch.save(checkpoint, os.path.join(PATH, 'epoch-{}.pth'.format(epoch+1)))
                
                    
