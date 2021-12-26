import torch
import torch.nn as nn
import torch.functional as F
from reward_model import RewardModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import ijson

"""
tokenizer = AutoTokenizer.from_pretrained("SophieTr/results")
supervised_baseline = AutoModelForSeq2SeqLM.from_pretrained("SophieTr/results")

model = RewardModel(supervised_baseline, d_model=1280)
"""
#### Write the training loop here

# First import the json data into pandas dataframes
numbers = [3]
for num in numbers:
    filename = "batch" + str(num) + ".json"
    with open(filename, 'r') as f:
        objects = ijson.items(f, 'info.item')
        print(list(objects)[0])













