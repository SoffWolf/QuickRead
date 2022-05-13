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

RUN_NAME = "PP0_rm_v1_full"#"ppo-peg-7e05-rm-1epoch_v3"#"PP0_rm_v1"
PATH = "./" + RUN_NAME
CHECKPOINT_PATH = os.path.join(PATH, 'latest_epo.pth')#'epoch-8.pth')#'epoch-16.pth') #'latest_minibatch.pth')

supervised_baseline = PegasusForConditionalGeneration.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")
tokenizer = PegasusTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")

# Policy
policy = PegasusWithValueHead(supervised_baseline)
policy.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH), map_location=torch.device('cpu')), strict=False)

# Upload to HuggingFace hub
# with Repository("ppo-model", clone_from="QuickRead/PP0_rm_v1_full", use_auth_token=True).commit(commit_message="PPO demo model :)"):
#     torch.save(policy.state_dict(), "model.pt")
test_data = """
As is customary with treatments of relational database management, we may use
the terms “database” and “relation” with two meanings: on the one hand, a logical
database (resp. a relation) may mean a database value (resp. relation value), that
is, some fixed contents of the database (resp. relation) as a bag of tuples, and on
the other hand, it may mean a database variable (resp. relation variable), that is,
a variable that can take a database (resp. relation) as its value. The context should
make it clear which of the two meanings is assumed. Sometimes, when we wish to
emphasize the distinction between the two meanings, we may talk about database
states when referring to database values.
For each logical database and for each of its relations, considered with the latter
meaning (as a variable), there is a set of associated integrity constraints, which
have been specified at the time of creating the database or relation (with the SQL
create table statement) or added afterwards (with the SQL alter table statement).
An integrity constraint may be internal to a single relation such as a key constraint
(primary key, unique) or it may span over two relations such as a referential
integrity constraint (foreign key).
Integrity constraints restrict the values that the database and relation variables are
allowed to take. A logical database (i.e., its state) is integral or consistent if it fulfills
the integrity constraints specified for the database.
An attempt to perform an update action that violates an integrity constraint
specified by SQL on the database either returns with an error indication, or triggers
a sequence of corrective actions so as to make the violated constraint to hold
again if such corrective actions have been specified. In addition to constraints
that can be specified by SQL create table and alter table statements, the logical
database usually satisfies some application-specific constraints. Applications using
the database must check for these and keep them satisfied.
"""
query = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'RESPONSE from test_data is:\n {resp_txt}')
