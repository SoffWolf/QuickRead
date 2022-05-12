import torch
import os
import sys
sys.path.insert(0,'..')
# from datasets import load_dataset, load_from_disk

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from pegasus_with_heads import PegasusWithValueHead
from ppo import PPOTrainer
from rewards.reward_model import RewardModel



RUN_NAME = "PP0_rm_v1_full"#"ppo-peg-7e05-rm-1epoch_v3"#"PP0_rm_v1"
PATH = "./" + RUN_NAME
CHECKPOINT_PATH = os.path.join(PATH, 'latest_epo.pth')#'epoch-8.pth')#'epoch-16.pth') #'latest_minibatch.pth')

supervised_baseline = PegasusForConditionalGeneration.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")
tokenizer = PegasusTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")

# Policy
policy = PegasusWithValueHead(supervised_baseline)
policy.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH), map_location=torch.device('cpu')), strict=False)

# Data
# dataset = load_from_disk("../../../QuickRead/reddit_clean")
# train_texts, train_labels = dataset['train']['content'], dataset['train']['summary']
test_data = """
    Wayne Rooney cut a forlorn figure after being spotted filling up his £100,000 Overfinch Range Rover, just a couple of days after his side’s loss to Chelsea - despite having 70 per cent possession.

A solitary Eden Hazard strike in the first half was enough for Jose Mourinho’s side, who showed how to ‘park the bus’ with a terrific defensive display as Manchester United struggled to break them down and create any clear-cut chances.

Rooney didn’t look best pleased with the photographer as he scowled at his picture being taken.

Wayne Rooney doesn’t look best pleased after being spotted filling up his £100,000 Range Rover

The England captain was disappointed with the result on Saturday but thought his side deserved to get something from the game.

He said: ‘In terms of the way we moved the Chelsea players about, making them work, it was excellent.

’Over the last few months it has all started to click and the players understand what the manager wants. That’s showing in the performances.

’I’ve rarely seen a team come to Stamford Bridge and dominate so much. All that was missing was the goal.’

The 29-year-old looked glum just a couple of days after his side’s loss to Chelsea in the Premier League

The England captain was seen here driving his £100,000 Overfinch Range Rover to training at Carrington

Rooney was played in a deeper role against Chelsea but was unable to help his side claim any points
"""
query = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'RESPONSE from test_data is:\n {resp_txt}')