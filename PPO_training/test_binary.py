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
    A driver was shocked to receive a £50 fine for failing to display his tax disc - despite it not becoming a legal requirement six months ago.

Phil Haste, 60, was also told he had not shown a pay and display ticket in his car, even though he has a valid car parking permit on his dashboard.

The businessman is now refusing to pay the fine that was issued in a Torbay council car park.

Scroll down for video

’Diabolical’: Phil Haste (left) said he could not understand why he was issued with a penalty notice, as he clearly had a valid parking permit in his car and no longer required his tax disc to be shown

Mr Haste sent his appeal to the council on Saturday after being issued with the ticket last month and said it was a ‘diabolical’ decision to fine him.

The government abolished paper tax discs on October 1 last year, meaning they no longer need to be shown on a vehicle windscreen.

Mr Haste, who lives in Torquay, said: ‘I would rather go to court than pay the fine. Issuing tickets of this ilk show they haven’t got a clue what they are talking about.

’It’s diabolical. The council employ these people to do a job for them and it’s clearly not being presented clearly.

’They are obviously trying to just grab money where they can. I don’t understand why I got this fine.’

’Haven’t got a clue’: The 60-year-old was also told he had not shown a pay and display ticket in his car, even though he has a valid car parking permit on his dashboard (pictured)

The 60-year-old, who owns the yacht brokers’ Quayside Marine, added: ‘I don’t need a tax disc so I don’t know why they issued this.

’I have a car parking permit which was clearly on display on the dashboard of my car, so I just can’t understand it.’

Torbay Council said it was not their policy to issue fines for road tax offences and urged Mr Haste to appeal.

A spokesman said: ‘We do not issue parking penalty charge notices with regard to road tax and Mr Haste will have been advised on how to appeal against this ticket as it is stated on the reverse of the penalty charge notice.’
"""
query = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'RESPONSE from test_data is:\n {resp_txt}')