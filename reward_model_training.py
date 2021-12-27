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

train_iter = feedback_data.iterrows()

def yield_tokens(data_iter):
    for index, row in data_iter:
        row = list(row) 
        row = [post, split, summary1, summary2, choice]
        text = post + summary1 + summary2
        yield tokenizer(text)


print("start build vocab")
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
print("finish build vocab")
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    post_list, sum1_list, sum2_list, label_list, offsets1, offsets2 = [], [], [], [], [0], [0]
    for (_post, _split, _sum1, _sum2, _label) in batch:
        processed_post = torch.tensor(text_pipeline(_post), dtype=torch.int64)
        processed_sum1 = torch.tensor(text_pipeline(_post), dtype=torch.int64)
        processed_sum2 = torch.tensor(text_pipeline(_post), dtype=torch.int64)
        label_list.append(label_pipeline(_label))
        # offsets.append(processed_post.size(0))
        offsets1.append(processed_sum1.size(0))
        offsets2.append(processed_sum2.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets1 = torch.tensor(offsets1[:-1]).cumsum(dim=0)
    offsets2 = torch.tensor(offsets2[:-1]).cumsum(dim=0)
    post_list = torch.cat(post_list)
    sum1_list = torch.cat(sum1_list)
    sum2_list = torch.cat(sum2_list)
    return post_list.to(device), sum1_list.to(device), sum2_list.to(device), label_list.to(device), offsets1.to(device), offsets2.to(device)

print("collate")
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

num_class = 2
vocab_size = len(vocab)
emsize = 64
model = model.to(device)    




def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (post, sum1, sum2, label, offsets1, offsets2) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_reward_1 = 0
        predicted_reward_2 = 0
        if label == 0:
            predicted_reward_1 = model(post, sum1, offsets1)
            predicted_reward_2 = model(post, sum2, offsets2)
        else:
            predicted_reward_2 = model(post, sum1, offsets1)
            predicted_reward_1 = model(post, sum2, offsets2)
        loss = criterion(predicted_reward_1 - predicted_reward_2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

print("train")
def evaluate(dataloader): #???
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

print("evaluate")

# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

def criterion(x):
    return np.log(1/(1 + np.exp(-x)))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
# train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
# test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
# valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              #shuffle=True, collate_fn=collate_batch)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
print("before training loop")
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    # accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy '.format(epoch,
                                           time.time() - epoch_start_time,
                                           ))
    print('-' * 59)



