import torch
import os
import sys
sys.path.insert(0,'..')
# from datasets import load_dataset, load_from_disk

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from pegasus_with_heads import PegasusWithValueHead
from ppo import PPOTrainer
from rewards.reward_model import RewardModel

from huggingface_hub import HfApi, create_repo, Repository

# create_repo("QuickRead/PP0_rm_v1_full")

RUN_NAME = "PPO_v8" #"PP0_rm_v1_full"#"ppo-peg-7e05-rm-1epoch_v3"#"PP0_rm_v1"
PATH = "./" + RUN_NAME
CHECKPOINT_PATH = os.path.join(PATH, 'latest_minibatch.pth') #'latest_epo.pth')#'epoch-8.pth')#'epoch-16.pth') #'latest_minibatch.pth')

supervised_baseline = PegasusForConditionalGeneration.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")
tokenizer = PegasusTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")

# Policy
policy = PegasusWithValueHead(supervised_baseline)
policy.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH), map_location=torch.device('cpu')), strict=False)

# Upload to HuggingFace hub
# with Repository("ppo-model", clone_from="QuickRead/PP0_rm_v1_full", use_auth_token=True).commit(commit_message="PPO demo model :)"):
#     torch.save(policy.state_dict(), "model.pt")
test_data_tifu = """
A bit of background about me. I'm 22 years old and I live in a suburban/rural town in NY. I grew up in New York City but ended up leaving several years ago due to college and family circumstances. I wanted to go to college, just not as far as I ended up going. Well anyways I'm no longer in college but I'm still in this town. And throughout the years I've let depression win over me. I ended up working in a local grocery store doing a dead-end job, which I quit because of inadequate management.

I decided it was best for me to not be here; I do miss the little bit of family I do have in my home city. So I got up one day and started looking for remote/hybrid positions located there. I ended up getting looked at by a really good company located in the heart of the city. One thing leads to another and they end up offering me a full-time position with them that actually starts pretty soon. I accepted the position.

However, I'm almost completely out of money. I don't have a solid way there nor do I have a solid place to stay. I don't know why I said yes. It honestly felt too good to be true, it probably is. And I feel terrible because I feel like I'm wasting the company's time.
"""
query = tokenizer(test_data_tifu, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt_tifu = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'\nRESPONSE from test_data from r/TIFU is:\n {resp_txt_tifu}')


test_data_news = """
Elon Musk said he is putting his bid to acquire Twitter (TWTR) on hold, weeks after agreeing to take the company private in a $44 billion deal.
"Twitter deal temporarily on hold pending details supporting calculation that spam/fake accounts do indeed represent less than 5% of users," Musk tweeted on Friday.
The news initially sent Twitter shares down more than 20% in premarket trading before the stock rebounded somewhat. Two hours after his first tweet, Musk posted that he is "still committed to acquisition."

In his tweet about putting the deal on hold, Musk linked to a May 2 Reuters report about Twitter's most recent disclosure about its spam and fake account problem.

But it acknowledged that the measurements were not independently verified and the actual number of fake or spam accounts could be higher.
Twitter has had a spam problem for years, and the company has previously acknowledged that reducing fakeIn its quarterly financial report, released on April 28, Twitter estimated that fake or spam accounts made up fewer than 5% of the platform's active users during the first three months of the year. Twitter noted that the estimates were based on a review of sample accounts and it believed the numbers to be "reasonable."
 and malicious accounts would play a key factor in its ability to keep growing. It's unclear why Musk would back away from the deal because of the latest disclosure.
"""
query = tokenizer(test_data_news, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt_news = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'\nRESPONSE from test_data from CNN news is:\n {resp_txt_news}')

test_data_article = """
Students are often tasked with reading a document and producing a summary (for example, a book report) to demonstrate both reading comprehension and writing ability. This abstractive text summarization is one of the most challenging tasks in natural language processing, involving understanding of long passages, information compression, and language generation. The dominant paradigm for training machine learning models to do this is sequence-to-sequence (seq2seq) learning, where a neural network learns to map input sequences to output sequences. While these seq2seq models were initially developed using recurrent neural networks, Transformer encoder-decoder models have recently become favored as they are more effective at modeling the dependencies present in the long sequences encountered in summarization. Transformer models combined with self-supervised pre-training (e.g., BERT, GPT-2, RoBERTa, XLNet, ALBERT, T5, ELECTRA) have shown to be a powerful framework for producing general language learning, achieving state-of-the-art performance when fine-tuned on a wide array of language tasks. In prior work, the self-supervised objectives used in pre-training have been somewhat agnostic to the down-stream application in favor of generality; we wondered whether better performance could be achieved if the self-supervised objective more closely mirrored the final task. In “PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization” (to appear at the 2020 International Conference on Machine Learning), we designed a pre-training self-supervised objective (called gap-sentence generation) for Transformer encoder-decoder models to improve fine-tuning performance on abstractive summarization, achieving state-of-the-art results on 12 diverse summarization datasets. Supplementary to the paper, we are also releasing the training code and model checkpoints on GitHub.
"""
query = tokenizer(test_data_article, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt_article = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'\nRESPONSE from test_data from Pegasus article is:\n {resp_txt_article}')
