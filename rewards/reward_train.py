import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from torch.optim import Adam
from transformers import PegasusTokenizer, PegasusModel #AutoTokenizer, AutoModelForSeq2SeqLM
from reward_model import RewardModel

## Global variables
# TODO: how to make this a param for runnig code??
RUN_NAME = "RM_incr_lr_v1"
SUPERVISED_MODEL = "QuickRead/pegasus-reddit-7e05"
EPOCHS = 1
LR = 1e-5
BATCH_SIZE = 1
# unchanged
DATAPATH = 'data/human_feedback.parquet'
KEY_PATH = "../PPO_training/hfAPI.txt"
CHECKPOINT_PATH = "./" + RUN_NAME

# -------------------------------------------------------------------------------- #
# if there is a failed train, trace the path to last saved checkpoint and add here #
# -------------------------------------------------------------------------------- #
PATH = CHECKPOINT_PATH +  "/lateststep.pth"

## Set up device

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

### DATA & DATALOADER related

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.post = [tokenizer(post, return_tensors="pt")['input_ids'] for post in df['post']]
        self.split = [split for split in df['split']]
        self.summary1 = [tokenizer(summary1, return_tensors="pt")['input_ids'] for summary1 in df['summary1']]
        self.summary2 = [tokenizer(summary2, return_tensors="pt")['input_ids'] for summary2 in df['summary2']]
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

# Purpose: Prevent duplicated training data tensor (Source: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/)
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

## TRAINING LOOP
def train(model, train_data, val_data, optimizer, resume=False, checkpoints={}):

    def criterion(x):
        s = nn.Sigmoid()
        sigmoid_r = s(x)

        # Criterion
        ret = torch.log(sigmoid_r)
        m = nn.LogSigmoid()
        ret = m(x) * - 1
        ret = torch.mean(ret)
        return ret

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    val_dataloader = torch.utils.data.DataLoader(val, collate_fn=collate, batch_size=BATCH_SIZE)

    if use_cuda:
        model = model.cuda()


    # WANDB 
    wandb.watch(model, log="all")
    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        acc_per_100 = 0
        step = 0
        if resume:
            if wandb.run.resumed:
                print("Resumed result from WandB")
            step = checkpoints['step']
            batch_loss = checkpoints['batch-loss']
            total_loss_train = checkpoints['total-batch-loss']
            total_acc_train = checkpoints['total-batch-acc']
            acc_per_100 = checkpoints['batch-total_acc_train-per-100-step']
        model.train(True)
        for post, sum1, sum2 in tqdm(train_dataloader):
            # Input
            post_id = post.to(device)
            sum1_id = sum1.to(device)
            sum2_id = sum2.to(device)
            
            # Zero gradients:
            optimizer.zero_grad()

            # Output rewards
            predicted_reward_1 = model(post_id, sum1_id)
            predicted_reward_2 = model(post_id, sum2_id)
            

            # Loss and accuracy
            batch_loss = criterion(torch.sub(predicted_reward_1,predicted_reward_2))
            total_loss_train += batch_loss
            step += 1
            
            # ACC increases when predicted_reward_1 is larger than predicted_reward_2 ??? 
            acc = (predicted_reward_1 > predicted_reward_2).sum().item()
            total_acc_train += acc
            acc_per_100 += acc

            # Backward
            batch_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                acc_per_100 = acc_per_100/(BATCH_SIZE * 100)
                # Logging
                wandb.log({ "train/batch-loss": batch_loss,
                            "train/total-batch-loss": total_loss_train,
                            "train/total-batch-acc": total_acc_train,
                            "train/batch-total_acc_train-per-100-step": acc_per_100})
                acc_per_100 = 0
            
            # Save checkpoint on every 10000 steps:
            if step % 10000 == 0:
                checkpoint = {
                        'step': step,
                        'state_dict': model.state_dict(),
                        'optimizer' :optimizer.state_dict(),
                        "batch-loss": batch_loss,
                        "total-batch-loss": total_loss_train,
                        "total-batch-acc": total_acc_train,
                        "batch-total_acc_train-per-100-step": acc_per_100
                        }
                torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, 'lateststep.pth'))
                wandb.save(os.path.join(CHECKPOINT_PATH, 'lateststep.pth'))

            # Manually update learning rate:
            if step % (100*700) == 0:
                print("Step where the learning rate is changed from 1e-6 to 9e-7: ", step)
                print("Previous LR = ", optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = 6e-6
                print("LR after updated = ", optimizer.param_groups[0]['lr'],"\n-------------------------------\n")
            
            if step % (100*1000) == 0:
                print("Step where the learning rate is changed from 9e-7 to 7e-7: ", step)
                print("Previous LR = ", optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = 5e-6
                print("LR after updated = ", optimizer.param_groups[0]['lr'],"\n-------------------------------\n")
            
            if step % (100*1300) == 0:
                print("Step where the learning rate is changed from 7e-7 to 5e-7: ", step)
                print("Previous LR = ", optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = 3e-6
                print("LR after updated = ", optimizer.param_groups[0]['lr'],"\n-------------------------------\n")
            
            if step % (100*1400) == 0:
                print("Step where the learning rate is changed from 5e-7 to 2e-7: ", step)
                print("Previous LR = ", optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = 1e-6
                print("LR after updated = ", optimizer.param_groups[0]['lr'],"\n-------------------------------\n")

        total_acc_val = 0
        total_loss_val = 0
        step = 0
        acc_per_100 = 0
        model.train(False)
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
                    acc_per_100 = acc_per_100/(BATCH_SIZE * 100)
                    
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

        # Save model at the end of an epoch
        checkpoint = {
                    'epoch': epoch_num+1,
                    'state_dict': model.state_dict(),
                    'optimizer' :optimizer.state_dict(),
                    'train_loss' : total_loss_train / len(train_data),
                    'val_loss' : total_loss_val / len(val_data)
                    }
        torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, 'epoch-{}.pth'.format(epoch_num+1)))

    model.push_to_hub("QuickRead/" + RUN_NAME)
    tokenizer.push_to_hub("QuickRead/" + RUN_NAME)
    

def test(model, df_test):
    if use_cuda:
        model = model.cuda()
    test = Dataset(df_test)

    test_dataloader = torch.utils.data.DataLoader(test, collate_fn=collate)
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

if __name__== "__main__":

    ### Read df from file & split.
    df = pd.read_parquet(DATAPATH, engine="pyarrow")
    # TODO: to save more memory, it is possible to split train, val, test beforehand and save to file. 
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df)), int(.95*len(df))])

    ## WANDB INIT
    group = "quickread"
    project = "text-summary-reward-model"
    display_name = RUN_NAME
    wandb.init(entity=group, project=project, name=display_name, resume=True)

    ### Load model
    tokenizer = PegasusTokenizer.from_pretrained(SUPERVISED_MODEL)
    supervised_baseline = PegasusModel.from_pretrained(SUPERVISED_MODEL)

    model = RewardModel(supervised_baseline)
    optimizer = Adam(model.parameters(), lr= LR)
    # Add logic for checking if thhere are checkpoints:
    try:
        # make path
        os.mkdir(RUN_NAME)
        save_directory = "QuickRead/" + RUN_NAME
        #model.save(save_directory, True, key, "QuickRead")

        train(model, df_train, df_val, optimizer)

    except:
        # load check points 
        print("Resumed training from checkpoint")
        
        checkpoint = torch.load(PATH)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        train(model, df_train, df_val, optimizer, resume=True, checkpoints=checkpoint)
        # model.to(device)
        
        # model.train()

    finally:
        test(model, df_test)

#https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html#load-the-general-checkpoint 
#https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 



