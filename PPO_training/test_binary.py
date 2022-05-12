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
I'm currently a junior doctor in my first few weeks on an Emergency Department rotation.

Today a teenage girl came in having fallen off her skateboard, cutting open her forehead. Normal procedure would either be for a doctor to apply sutures, or a nurse / medical technician with experience in gluing to effectively use what is superglue to seal the skin together - of course neither were available on my shift during one of the busiest parts of the day.

I would have been more comfortable suturing the laceration given we were actually taught how to do this at medical school. Gluing - not so much. My superior advised me not to suture the laceration as I haven't done any suturing for a while, and given that the laceration was on the face, to either wait for someone with more experience to do it, or glue it with care and attention.

The patient had of course already been waiting hours to see a doctor and absolutely did not want to wait any longer. Being the absolute hero I am, I decided to give gluing a go. I wondered how hard it could actually be given I'm more than proficient in supergluing the soles of my trainers together when they fall apart on a weekly basis.

The choice of glue is cyanoacrylate - a glue that I am told by another equally bold junior doctor would not be able to glue surfaces other than skin together, and therefore perfectly safe to use copious amounts of. I tested this theory out by applying glue to my gloved index finger and thumb, and trying to glue them together. Nothing happened - my finger and thumb came apart as if I was using water.

So on I went and started to apply glue to this young girl's forehead whilst holding the skin together tightly. Only it turns out that actually any surface with moisture on it is enough to activate the cyanoacrylate, and the moisture in skin is what causes it to stick together. When I tested out the glue on the gloves earlier, I had just applied new dry gloves, so there was nothing to activate the glue, and now the gloves were likely covered in blood and moisture from manipulating the laceration into position and were a fully bondable surface."""
query = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'RESPONSE from test_data is:\n {resp_txt}')