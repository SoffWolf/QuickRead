import torch
import os
import sys
sys.path.insert(0,'..')
# from datasets import load_dataset, load_from_disk

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from pegasus_with_heads import PegasusWithValueHead
from ppo import PPOTrainer
from rewards.reward_model import RewardModel



RUN_NAME = "PP0_rm_v1_full"#"ppo-peg-7e05-rm-1epoch_v3"#"PP0_rm_v1"
PATH = "./" + RUN_NAME
CHECKPOINT_PATH = os.path.join(PATH, 'latest_epo.pth')#'epoch-8.pth')#'epoch-16.pth') #'latest_minibatch.pth')

supervised_baseline = PegasusForConditionalGeneration.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")
tokenizer = PegasusTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")

# Policy
policy = PegasusWithValueHead(supervised_baseline)
policy.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH), map_location=torch.device('cpu')), strict=False)

# Data
# dataset = load_from_disk("../../../QuickRead/reddit_clean")
# train_texts, train_labels = dataset['train']['content'], dataset['train']['summary']
test_data = """
    Finland continues to move towards membership of NATO. Ukrainian President Volodymyr Zelensky said he commends Finland’s "readiness" to join NATO, while the Kremlin said it would see Finland's accession to the military alliance as a threat.

On the ground, all the civilians are believed to have been evacuated from Mariupol’s Azovstal steel plant.

Here are the latest updates from Russia’s invasion of Ukraine:

Finland’s NATO membership: Finland's leaders announced in a joint statement on Thursday that they are in favor of applying for NATO membership, moving the Nordic nation closer to joining the alliance. Since the Russian invasion of Ukraine, public support for joining NATO in Finland, which shares an 800-mile border with Russia, has leaped from around 30% to nearly 80% in some polls.

Sweden could be next: It is also expected that Sweden, Finland’s neighbor to the west, will soon announce its intention to join NATO. Sweden's foreign minister said Thursday that the country will "take Finland’s assessments into account." Russia has warned both countries against joining the alliance, saying there would be consequences.

Support for Finland: NATO chief Jens Stoltenberg said Finland would be "warmly welcomed" into the alliance. Meanwhile, NATO members Denmark and Estonia said they would support Finland’s membership, with Danish Prime Minister Mette Frederiksen saying it "will strengthen NATO and our common security."

Moscow's reaction: Kremlin spokesman Dmitry Peskov said Thursday that Russia would see Finland's accession to the NATO as a threat and the move would not contribute to more security. Russia will analyze the situation with Finland's entry to NATO and will work out the necessary measures to ensure its own security, he added.
"""
query = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'RESPONSE from test_data is:\n {resp_txt}')