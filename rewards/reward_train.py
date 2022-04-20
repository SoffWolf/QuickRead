import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.utils.rnn import pad_sequence
from reward_model import RewardModel
from transformers import PegasusTokenizer, PegasusModel #AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data.dataset import random_split
import time
import ijson
import pandas as pd
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import wandb
import os


# First import the json data into pandas dataframes
numbers = [i+3 for i in range (18)] + [22]
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
    filename = "reward_training_data/batch" + str(num) + ".json"
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.post = [tokenizer(post, return_tensors="pt")['input_ids'] for post in df['post']]
        self.split = [split for split in df['split']]
        self.summary1 = [tokenizer(summary1, return_tensors="pt")['input_ids'] for summary1 in df['summary1']]
        self.summary2 = [tokenizer(summary2, return_tensors="pt")['input_ids'] for summary2 in df['summary2']]# padding='max_length',
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
        batch_sum1 = self.get_batch_sum1(idx)
        batch_sum2 = self.get_batch_sum2(idx)
        return batch_post, batch_sum1, batch_sum2

np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df)), int(.95*len(df))])

tokenizer = PegasusTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05")
supervised_baseline = PegasusModel.from_pretrained("QuickRead/pegasus-reddit-7e05") # Tobechange

model = RewardModel(supervised_baseline)
keys_file = open("../PPO_training/hfAPI.txt")
key = keys_file.readlines()[0].rstrip()

### TO BE UNCOMMENT AFTER DEBUG
#save_directory = "QuickRead/Reward_training_Pegasus_reddit"
#model.save(save_directory, True, 'https://huggingface.co/QuickRead/Reward_training_Pegasus_reddit', key, "QuickRead")


#Collate
def collate(list_of_samples):
    """Merges a list of samples to form a mini-batch."""
    posts = []
    post_mask = []
    sum1s = []
    sum2s = []
    # paste data to src_seqs, tgt_seqs
    for i in list_of_samples:
        posts.append(torch.squeeze(i[0],0))
        sum1s.append(torch.squeeze(i[1],0))
        sum2s.append(torch.squeeze(i[2],0))
        #print("done appending: ", i[0].shape, "\n sum1: ", i[1].shape, "\n sum2: ", i[2].shape)
    # zero-padding 
    posts = pad_sequence(posts, True, padding_value=-1)
    sum1s = pad_sequence(sum1s, True, padding_value=-1)
    sum2s = pad_sequence(sum2s, True, padding_value=-1)
    #print("Done padding: ", posts.shape, sum1s.shape, sum2s.shape)
    for i in posts:
        row = []
        for j in i:
            if j==-1:
                row.append(1)
            else: 
                row.append(0)
        post_mask.append(row)
    post_mask = torch.BoolTensor(post_mask)                 # tranform to right dtype: torch.bool
    return posts, sum1s, sum2s


# WANDB 
# import wandb

# WANDB 
user = "sophietr"
group = "quickread"
project = "text-summary-reward-model"
display_name = "reward_model_wandb_7e5_bs_1_idx"
wandb.init(entity=group, project=project, name=display_name)


# training loop
def train(model, train_data, val_data, learning_rate, epochs, bs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    def criterion(x):
        # For tracking purposes: --> DELETE later
        #print("\n x =", x)
        s = nn.Sigmoid()
        sigmoid_r = s(x)
        #print("\n Sigmoid = ", sigmoid_r)

        # Criterion
        ret = torch.log(sigmoid_r)
        m = nn.LogSigmoid()
        ret = m(x) * - 1
        ret = torch.mean(ret)
        #print("\n ret from criterion = ", ret)
        return ret

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=bs, collate_fn=collate, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, collate_fn=collate, batch_size=bs)

    if use_cuda:
        model = model.cuda()

    optimizer = Adam(model.parameters(), lr= learning_rate)

    # WANDB 
    wandb.watch(model, log="all")
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        acc_per_100 = 0
        step = 0
        model.train()
        for post, sum1, sum2 in tqdm(train_dataloader):
            # Input
            post_id = post.to(device)
            sum1_id = sum1.to(device)
            sum2_id = sum2.to(device)
            
            # Output rewards
            predicted_reward_1 = model(post_id, sum1_id)
            predicted_reward_2 = model(post_id, sum2_id)
            optimizer.zero_grad()

            # Loss and accuracy
            batch_loss = criterion(torch.sub(predicted_reward_1,predicted_reward_2))
            total_loss_train += batch_loss
            step += 1
            
            # ACC increases when predicted_reward_1 is larger than predicted_reward_2 ??? 
            acc = (predicted_reward_1 > predicted_reward_2).sum().item()
            total_acc_train += acc
            acc_per_100 += acc

            # Backward
            # model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                acc_per_100 = acc_per_100/(bs * 100)
                # Logging
                wandb.log({ "train/batch-loss": batch_loss,
                            "train/total-batch-loss": total_loss_train,
                            "train/total-batch-acc": total_acc_train,
                            "train/batch-total_acc_train-per-100-step": acc_per_100})
                acc_per_100 = 0

            if step % (100*1000) == 0:
                print("Step where the learning rate is changed from 1e-6 to 9e-7: ", step)
                print("Previous LR = ", optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = 9e-7
                print("LR after updated = ", optimizer.param_groups[0]['lr'],"\n-------------------------------\n")
            
            if step % (100*1200) == 0:
                print("Step where the learning rate is changed from 9e-7 to 7e-7: ", step)
                print("Previous LR = ", optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = 7e-7
                print("LR after updated = ", optimizer.param_groups[0]['lr'],"\n-------------------------------\n")
            
            if step % (100*1400) == 0:
                print("Step where the learning rate is changed from 7e-7 to 5e-7: ", step)
                print("Previous LR = ", optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = 5e-7
                print("LR after updated = ", optimizer.param_groups[0]['lr'],"\n-------------------------------\n")
            
            if step % (100*1500) == 0:
                print("Step where the learning rate is changed from 5e-7 to 2e-7: ", step)
                print("Previous LR = ", optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = 2e-7
                print("LR after updated = ", optimizer.param_groups[0]['lr'],"\n-------------------------------\n")
                
        total_acc_val = 0
        total_loss_val = 0
        step = 0
        acc_per_100 = 0
        model.eval()
        with torch.no_grad():

            for post, sum1, sum2 in tqdm(val_dataloader):

                # Input
                post_id = post.to(device)
                sum1_id = sum1.to(device)
                sum2_id = sum2.to(device)

                #Output rewards
                predicted_reward_1 = model(post_id, sum1_id)
                predicted_reward_2 = model(post_id, sum2_id)

                # Loss and accuracy
                batch_loss = criterion(torch.sub(predicted_reward_1,predicted_reward_2))
                total_loss_val+= batch_loss
                step += 1

                acc = (predicted_reward_1 > predicted_reward_2).sum().item()                
                total_acc_val += acc
                acc_per_100 += acc
                if step % 100 == 0:
                    acc_per_100 = acc_per_100/(bs * 100)
                    
                    # Logging
                    wandb.log({ "val/batch-loss": batch_loss,
                                "val/total-batch-loss": total_loss_val,
                                "val/total-batch-acc": total_acc_val,
                                "val/batch-total_acc-per-100-step": acc_per_100})
                    acc_per_100 = 0
        print(
              f'Epochs: {epoch_num + 1} \
              | Train Loss: {total_loss_train / len(train_data): .3f} \
              | Train Accuracy: {total_acc_train / len(train_data): .3f} \
              | Val Loss: {total_loss_val / len(val_data): .3f} \
              | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        
        wandb.log({"Epoch": epoch_num + 1,
                   "train/Epoch-train-loss": total_loss_train / len(train_data),
                   "train/Train-acc": total_acc_train / len(train_data),
                   "val/Epoch-val-loss": total_loss_val / len(val_data),
                   "val/Val-acc": total_acc_val / len(val_data) })

    # Save model
    checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
    torch.save(checkpoint, os.path.join("./reward_model_wandb_7e5_bs_1_idx", 'epoch-{}.pth'.format(epoch_num+1)))

    model.push_to_hub("QuickRead/Reward_training_Pegasus_reddit")
    tokenizer.push_to_hub("QuickRead/Reward_training_Pegasus_reddit")
    
    return model

def test(model, df_test):
    test_dataloader = torch.utils.data.DataLoader(df_test, collate_fn=collate)
    total_acc_test = 0
    step = 0
    acc_per_100 = 0
    model.eval()
    with torch.no_grad():
        for post, sum1, sum2 in tqdm(test_dataloader):

            # Input
            post_id = post.to(device)
            sum1_id = sum1.to(device)
            sum2_id = sum2.to(device)

            #Output rewards
            predicted_reward_1 = model(post_id, sum1_id)
            predicted_reward_2 = model(post_id, sum2_id)

            acc = (predicted_reward_1 > predicted_reward_2).sum().item()                
            total_acc_test += acc
            acc_per_100 += acc
            if step % 100 == 0:
                acc_per_100 = acc_per_100/(100)
            acc_per_100 = 0
    print(
            f'Test accuracy: {total_acc_test / len(df_test): .3f}\n')

EPOCHS = 1
LR = 1e-6
BATCH_SIZE = 1
trained_model = train(model, df_train, df_val, LR, EPOCHS, BATCH_SIZE)
test(trained_model, df_test)





