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
summary1 = ""
summary2=""
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
                elif len(summaries) < 2:
                    summary2 = value
                    summaries.append(summary2)
            elif (prefix, event) == ("choice", "number"):
                choice = value
                if choice == 1:
                    temp = summary1
                    summary1 = summary2
                    summary2 = temp
                chosen_row.append(summary1)
                chosen_row.append(summary2)
                chosen_row.append(choice)
                data.append(chosen_row)
                # Reset
                chosen_row = []


df = pd.DataFrame(data, columns=columns)
tokenizer = AutoTokenizer.from_pretrained("SophieTr/fine-tune-Pegasus")
supervised_baseline = AutoModelForSeq2SeqLM.from_pretrained("SophieTr/fine-tune-Pegasus")



class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.post = [tokenizer(post, return_tensors="pt") for post in df['post']]
        self.split = [split for split in df['split']] 
        self.summary1 = [tokenizer(summary1, return_tensors="pt") for summary1 in df['summary1']]
        self.summary2 = [tokenizer(summary2, return_tensors="pt") for summary2 in df['summary2']]# padding='max_length', 
        self.labels = [label for label in df['choice']]
    def classes(self):
        return self.labels
    def __len__(self):
        return len(self.labels)
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(int(self.labels[idx]))
    def get_batch_posts(self, idx):
        # Fetch a batch of inputs
        return self.post[idx]
    def get_batch_split(self, idx):
        # Fetch a batch of inputs
        return self.split[idx]
    def get_batch_sum1(self, idx):
        # Fetch a batch of inputs
        return self.summary1[idx]
    def get_batch_sum2(self, idx):
        # Fetch a batch of inputs
        return self.summary2[idx]
    def __getitem__(self, idx):
        batch_post = self.get_batch_posts(idx)
        batch_split = self.get_batch_split(idx)
        batch_sum1 = self.get_batch_sum1(idx)
        batch_sum2 = self.get_batch_sum2(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_post, batch_split, batch_sum1, batch_sum2, batch_labels


np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df)), int(.95*len(df))])

print("Split the examples for train, val, test: ", len(df_train),len(df_val), len(df_test))


model = RewardModel(supervised_baseline)                  

# training loop
def train(model, train_data, val_data, learning_rate, epochs):

    def criterion(x):
        return torch.log(1/(1 + torch.exp(-x)))

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for post, split, sum1, sum2, label in tqdm(train_dataloader):
            post_id = post['input_ids'].squeeze(1).squeeze(1)
            sum1_id = sum1['input_ids'].squeeze(1).squeeze(1)
            sum2_id = sum2['input_ids'].squeeze(1).squeeze(1)

                    
            label = label.to(device)

            predicted_reward_1 = model(post_id, sum1_id)
            predicted_reward_2 = model(post_id, sum2_id)
                    
            
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
                post_id = post['input_ids'].squeeze(1).squeeze(1)
                sum1_id = sum1['input_ids'].squeeze(1).squeeze(1)
                sum2_id = sum2['input_ids'].squeeze(1).squeeze(1)
                        
                label = label.to(device)

                predicted_reward_1 = model(post_id, sum1_id)
                predicted_reward_2 = model(post_id, sum2_id)
                         
                batch_loss = criterion(predicted_reward_1 - predicted_reward_2)
                total_loss_val += batch_loss.item()

                
                # acc = (output.argmax(dim=1) == val_label).sum().item()
                # total_acc_val += acc
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        torch.save(model, os.path.join("./reward_model_weight", 'epoch-{}.pth'.format(epoch_num)))

EPOCHS = 5
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)





