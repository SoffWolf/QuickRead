import torch
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
sys.path.insert(0,'..')
from pathlib import Path  
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM


from pegasus_with_heads import PegasusWithValueHead
from ppo import PPOTrainer
from rewards.reward_model import RewardModel

from huggingface_hub import HfApi, create_repo, Repository

supervised = True
#### IMPORT DATA
DATAPATH = '../rewards/data/human_feedback.parquet'
df = pd.read_parquet(DATAPATH, engine="pyarrow")
n_sample = 10
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df)), int(.95*len(df))])

input_posts = list(df_test['post'].values)
label_summaries_1 = list(df_test['summary1'].values)
label_summaries_2 = list(df_test['summary2'].values)


#### IMPORT MODEL

RUN_NAME = "finetuned_Pegasus" #"PPO_v8" #"PP0_rm_v1_full"#"ppo-peg-7e05-rm-1epoch_v3"#"PP0_rm_v1"

### OUTPUT PATH
OUTPUT_NAME = RUN_NAME + '_out.parquet'
OUT_PATH = str(Path("test_binary.py").parent)+'/ppo_output/'+OUTPUT_NAME


model_name = "QuickRead/pegasus-reddit-7e05"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

if not supervised:
    PATH = "./" + RUN_NAME
    CHECKPOINT_PATH = os.path.join(PATH, 'latest_minibatch.pth') #'latest_epo.pth')#'epoch-8.pth')#'epoch-16.pth') #'latest_minibatch.pth')
    supervised_baseline = PegasusForConditionalGeneration.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")
    tokenizer = PegasusTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")
    policy = PegasusWithValueHead(supervised_baseline)
    policy.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH), map_location=torch.device('cpu')), strict=False)


# Upload to HuggingFace hub
# with Repository("ppo-model", clone_from="QuickRead/PP0_rm_v1_full", use_auth_token=True).commit(commit_message="PPO demo model :)"):
#     torch.save(policy.state_dict(), "model.pt")

## Model generates: 
def preprocess(inp):
    input_ids = tokenizer(inp, padding=True, truncation=True, return_tensors='pt').input_ids
    return input_ids
def predict(input_ids, supervised = False):
    if supervised:
        outputs = model.generate(input_ids=input_ids)
    else:
        outputs = policy.generate(input_ids)
    
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return res

data = []
columns = [
    "post",
    "summary"
]
# for post in input_posts:
for i in tqdm(range(len(input_posts[:1500]))):
    post = input_posts[i]
    curr_row = []
    #print("\ni = ", i, "\n", post, flush=True)
    #print("===> Summary from model", flush=True)
    tokens = preprocess(post)

    response = predict(tokens, supervised)
    curr_row.append(post)
    curr_row.append(response)
    data.append(curr_row)
    #print(response, flush=True)
    #print("------------------------------->_<-------------------------------", flush=True)

df = pd.DataFrame(data, columns=columns)
df.to_parquet(OUT_PATH, engine="pyarrow", index=False)
print("Successfully create parquet file")
df = pd.read_parquet(OUT_PATH, engine="pyarrow")
print(df.dtypes)
print(df)
