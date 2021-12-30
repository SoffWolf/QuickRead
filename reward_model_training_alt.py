import torch
import torch.nn as nn
import torch.functional as F
from reward_model import RewardModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data.dataset import random_split
import time
import ijson
import pandas as pd
import numpy as np
from torch.optim import Adam
from tqdm import tqdm


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


df = pd.DataFrame(data, columns=columns)
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distill-pegasus-xsum-16-4")
supervised_baseline = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distill-pegasus-xsum-16-4")



class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.post = [tokenizer(post, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for post in df['post']]
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
    def get_batch_posts(self, idx):
        # Fetch a batch of inputs
        return self.post[idx]
    def get_batch_sum1(self, idx):
        # Fetch a batch of inputs
        return self.summary1[idx]
    def get_batch_sum2(self, idx):
        # Fetch a batch of inputs
        return self.summary2[idx]
    def __getitem__(self, idx):
        batch_post = self.get_batch_posts(idx)
        batch_sum1 = self.get_batch_sum1(idx)
        batch_sum2 = self.get_batch_sum2(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_post, batch_sum1, batch_sum2, batch_labels


np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df)), int(.95*len(df))])

print(len(df_train),len(df_val), len(df_test))

model = RewardModel(supervised_baseline)
post = """We use a 774M parameter version of the GPT-2 language
model in Radford et al. (2019) trained on their WebText
dataset and their 50,257 token invertible byte pair encoding
to preserve capitalization and punctuation (Sennrich et al.,
2015). The model is a Transformer with 36 layers, 20 heads,
and embedding size 1280 (Vaswani et al., 2017).
For stylistic continuation tasks we perform supervised finetuning of the language model to the BookCorpus dataset
of Zhu et al. (2015) prior to RL fine-tuning; we train from
scratch on WebText, supervised fine-tune on BookCorpus,
then RL fine-tune to our final task. To improve sample
quality, we use a temperature of T < 1 for all experiments;
we modify the initial language model by dividing logits by
T, so that future sampling and RL with T = 1 corresponds
to a lower temperature for the unmodified pretrained model."""
summary = "I want to do gymnastics, but I’m 28 yrs old. Is it too late for me to be a gymnaste?!"
post_id = tokenizer(post, return_tensors="pt").input_ids
sum_id = tokenizer(summary, return_tensors="pt").input_ids
print(model(post_id, sum_id))

# training loop
def train(model, train_data, val_data, learning_rate, epochs):

    def criterion(x):
        return np.log(1/(1 + np.exp(-x)))

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for post, split, sum1, sum2, label in tqdm(train_dataloader):
            # mask_post = post['attention_mask'].to(device)
            # mask_sum1 = sum1['attention_mask'].to(device)
            # mask_sum2 = sum2['attention_mask'].to(device)
            post_id = tokenizer(post)['input_ids'].squeeze(1).to(device)
            sum1_id = tokenizer(sum1)['input_ids'].squeeze(1).to(device)
            sum2_id = tokenizer(sum2)['input_ids'].squeeze(1).to(device)
                    
            label = label.to(device)
            predicted_reward_1 = None
            predicted_reward_2 = None

            if label == 0:
                predicted_reward_1 = model(post_id, sum1_id)
                predicted_reward_2 = model(post_id, sum2_id)
            else:
                predicted_reward_2 = model(post_id, sum1_id)
                predicted_reward_1 = model(post_id, sum2_id)
        
            
            batch_loss = criterion(predicted_reward_1 - predicted_reward_2)
            total_loss_train += batch_loss.item()
            
            # acc = (output.argmax(dim=1) == train_label).sum().item()
            # total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for post, split, sum1, sum2, label in tqdm(val_dataloader):
                post_id = tokenizer(post)['input_ids'].squeeze(1).to(device)
                sum1_id = tokenizer(sum1)['input_ids'].squeeze(1).to(device)
                sum2_id = tokenizer(sum2)['input_ids'].squeeze(1).to(device)
                        
                label = label.to(device)
                predicted_reward_1 = None
                predicted_reward_2 = None

                if label == 0:
                    predicted_reward_1 = model(post_id, sum1_id)
                    predicted_reward_2 = model(post_id, sum2_id)
                else:
                    predicted_reward_2 = model(post_id, sum1_id)
                    predicted_reward_1 = model(post_id, sum2_id)
        
            
                batch_loss = criterion(predicted_reward_1 - predicted_reward_2)
                total_loss_val += batch_loss.item()

                
                # acc = (output.argmax(dim=1) == val_label).sum().item()
                # total_acc_val += acc
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  
EPOCHS = 5
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)






