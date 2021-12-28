import torch
import torch.nn as nn
import torch.functional as F
from reward_model import RewardModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import time
import ijson
import pandas as pd

# First import the json data into pandas dataframes
numbers = [3]
data = []
columns = [
    "post",
    "split",
    "summary1",
    "summary2",
    "choice"
]

for num in numbers:
    filename = "batch" + str(num) + ".json"
    with open(filename, 'r') as f:
        parser = ijson.parse(f, multiple_values=True)
        chosen_row = []
        summaries = []
        for prefix, event, value in parser:
            if (prefix, event) == ("info.post", "string"):
                post = value
                chosen_row.append(post)
            elif (prefix, event) == ("split", "string"):
                split = value
                chosen_row.append(split)
            elif (prefix, event) == ("summaries.item.text","string"):
                if len(summaries) == 2:
                    summaries = []
                    summary1 = value
                    summaries.append(summary1)
                    chosen_row.append(summary1)
                elif len(summaries) < 2:
                    summary2 = value
                    summaries.append(summary2)
                    chosen_row.append(summary2)
            elif (prefix, event) == ("choice", "number"):
                choice = value
                chosen_row.append(choice)
                data.append(chosen_row)
                # Reset
                chosen_row = []


# Training loop
feedback_data = pd.DataFrame(data, columns=columns)
tokenizer = AutoTokenizer.from_pretrained("SophieTr/results")
supervised_baseline = AutoModelForSeq2SeqLM.from_pretrained("SophieTr/results")
model = RewardModel(supervised_baseline)

example_text = 'I will watch Memento tonight'
bert_input = tokenizer(example_text, padding='max_length', max_length = 10, truncation=True, return_tensors="pt")
print(bert_input)
print(bert_input['input_ids'])
print(bert_input['attention_mask'])

example_text = tokenizer.decode(bert_input.input_ids[0])
print(example_text)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.post = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['post']]
        self.split = [split for split in df['split']] 
        self.summary1 = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for summary1 in df['summary1']]
        self.summary2 = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for summary2 in df['summary2']]
        self.labels = [label for label in df['choice']]
    def classes(self):
        return self.labels
    def __len__(self):
        return len(self.labels)
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y












