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
#print(key)

### TO BE UNCOMMENT AFTER DEBUG
#save_directory = "QuickRead/Reward_training_Pegasus_reddit"
#model.save(save_directory, True, 'https://huggingface.co/QuickRead/Reward_training_Pegasus_reddit', key, "QuickRead")


#Collate
def collate(list_of_samples):
    """Merges a list of samples to form a mini-batch.

    Args:
      list_of_samples is a list of tuples (src_seq, tgt_seq):
          src_seq is of shape (src_seq_length)
          tgt_seq is of shape (tgt_seq_length)

    Returns:
      src_seqs of shape (max_src_seq_length, batch_size): LongTensor of padded source sequences.
      src_mask of shape (max_src_seq_length, batch_size): BoolTensor (tensor with boolean elements) indicating which
          elements of the src_seqs tensor should be ignored in computations: True values in src_mask correspond
          to padding values in src_seqs.
      tgt_seqs of shape (max_tgt_seq_length+1, batch_size): LongTensor of padded target sequences.
    """
    # YOUR CODE HERE
    posts = []
    post_mask = []
    sum1s = []
    sum2s = []
    print("into collate: ", list_of_samples[0])
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
    # transfor src_seqs to src_mask
    for i in posts:
        row = []
        for j in i:
            if j==0:
                row.append(1)
            else: 
                row.append(0)
        post_mask.append(row)
    post_mask = torch.BoolTensor(post_mask)                 # tranform to right dtype: torch.bool

    #sos1_ = torch.zeros(1, sum1s.shape[1])             # create sos_token with right shape
    #sos2_ = torch.zeros(1, sum2s.shape[1])             # create sos_token with right shape
    #sum1s = torch.cat((sos1_, sum1s), dim = 0).long() # insert to tgt_seqs as 1st row
    #sum2s = torch.cat((sos2_, sum2s), dim = 0).long() # insert to tgt_seqs as 1st row
    #print("et o et: ", sum1s.shape, sum2s.shape)
#     print(f'tgt_seqs:\n {tgt_seqs}')
#     print(f'src_seqs:\n {src_seqs}')
#     print(f'src_mask:\n {src_mask}')
    return posts, sum1s, sum2s


# WANDB 
# import wandb

# WANDB 
user = "sophietr"
group = "quickread"
project = "text-summary-reward-model"
display_name = "reward_model_wandb_7e5_bs_1"
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
        print("ret: ", ret)
        print("ret type: ", type(ret))
        ret = torch.mean(ret)
        print("after mean: ", ret)
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
    print("ENTERING FOR EPOCH LOOP") 
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        acc_per_100 = 0
        step = 0
        print("BEFORE TRAIN LOOP")
        for post, sum1, sum2 in tqdm(train_dataloader):
            # Input
            print(post, "\n", sum1, "\n", sum2)
            post_id = post.to(device)
            sum1_id = sum1.to(device)
            sum2_id = sum2.to(device)
            #print("TYPES: ", post_id.dtype, sum1_id.dtype, sum2_id.dtype)
            #print("SHAPES: ", post_id.shape, sum1_id.shape, sum2_id.shape)
            #post_id, sum1_id, sum2_id = post_id.to(device), sum1_id.to(device), sum2_id.to(device)
            print("SHAPES of post_id, sum1_id, sum2_id: ", post_id.shape, sum1_id.shape, sum2_id.shape)
            #try:
                # Output rewards
            
            print("GET after inputs, before MODEL call")
            predicted_reward_1 = model(post_id, sum1_id)
            predicted_reward_2 = model(post_id, sum2_id)
            print("predicted_reward_1: ", predicted_reward_1.shape,"\n", predicted_reward_1)
            print("predicted_reward_2: ",predicted_reward_2.shape, "\n", predicted_reward_2
)
            #except Exception as e: 
                #print("ERROR IN TRAIN LOOP (1)")
                #print(e)
                #print("SHAPES: ", post_id.shape, sum1_id.shape, sum2_id.shape)
                #continue
            optimizer.zero_grad()

            # Loss and accuracy
            batch_loss = criterion(torch.sub(predicted_reward_1,predicted_reward_2))
            print("batch_loss = ", batch_loss)
            total_loss_train += batch_loss
            step += 1
            # print("train batch loss: ", batch_loss)
            
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
                #   print("train batch total_acc_train/step: ", acc_per_100)
                # Logging
                wandb.log({ "train/batch-loss": batch_loss,
                            "train/total-batch-loss": total_loss_train,
                            "train/total-batch-acc": total_acc_train,
                            "train/batch-total_acc_train-per-100-step": acc_per_100})
                acc_per_100 = 0


        total_acc_val = 0
        total_loss_val = 0
        step = 0
        acc_per_100 = 0
        print("BEFORE VAL LOOP")
        with torch.no_grad():

            for post, sum1, sum2 in tqdm(val_dataloader):

                # Input
                post_id = post.to(device)
                sum1_id = sum1.to(device)
                sum2_id = sum2.to(device)

                #try:
                    # Output rewards
                predicted_reward_1 = model(post_id, sum1_id)
                predicted_reward_2 = model(post_id, sum2_id)
                    #print("predicted_reward_1: ", predicted_reward_1)
                    #print("predicted_reward_2: ",predicted_reward_2)
                #except: 
                    #print("ERROR IN TRAIN LOOP (2)")
                    #print("SHAPES: ", post_id.shape, sum1_id.shape, sum2_id.shape)
                    #continue

                # # Output rewards
                # predicted_reward_1 = model(post_id, sum1_id, device=device)
                # predicted_reward_2 = model(post_id, sum2_id, device=device)
                # print("predicted_reward_1: ", predicted_reward_1)
                # print("predicted_reward_2: ",predicted_reward_2)

                # Loss and accuracy
                batch_loss = criterion(torch.sub(predicted_reward_1,predicted_reward_2))
                total_loss_val+= batch_loss
                step += 1
                #print("eval batch loss: ", batch_loss)

                acc = (predicted_reward_1 > predicted_reward_2).sum().item()                
                total_acc_val += acc
                acc_per_100 += acc
                # # Logging
                # wandb.log({ "val/batch-loss": batch_loss,
                #             "val/total-batch-loss": total_loss_val,
                #             "val/total-batch-acc": total_acc_val,
                #             "val/batch-acc-per-step" :acc/step, 
                #             "val/batch-total_acc_val-per-step": total_acc_val/step})
                if step % 100 == 0:
                    acc_per_100 = acc_per_100/(bs * 100)
                    # print("val batch total_acc_val/100 step: ", acc_per_100)
                    
                    # Logging
                    wandb.log({ "val/batch-loss": batch_loss,
                                "val/total-batch-loss": total_loss_val,
                                "val/total-batch-acc": total_acc_val,
                                "val/batch-total_acc-per-100-step": acc_per_100})
                    acc_per_100 = 0

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f}\n')
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

        # Print model's state_dict
        #print("Model's state_dict:")
        #for param_tensor in model.state_dict():
            #print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        #print("Optimizer's state_dict:")
        #for var_name in optimizer.state_dict():
            #print(var_name, "\t", optimizer.state_dict()[var_name])
        
    # Save model
    checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
    # torch.save(model.state_dict(), PATH)
    torch.save(checkpoint, os.path.join("./reward_model_wandb_7e5_bs_1", 'epoch-{}.pth'.format(epoch_num+1)))

    # torch.save(model, os.path.join("./reward_model_weight_5ep", 'epoch-{}.pth'.format(epoch_num+1)))
    model.push_to_hub("QuickRead/Reward_training_Pegasus_reddit")
    tokenizer.push_to_hub("QuickRead/Reward_training_Pegasus_reddit")
    

EPOCHS = 1
LR = 1e-6
BATCH_SIZE = 1
train(model, df_train, df_val, LR, EPOCHS, BATCH_SIZE)





