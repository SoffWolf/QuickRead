import torch
import os

# from datasets import load_dataset, load_from_disk

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from pegasus_with_heads import PegasusWithValueHead
from ppo import PPOTrainer
from rewards.reward_model import RewardModel


config = {
    "lm_name": "QuickRead/pegasus-reddit-7e05",   # policy: supervised baseline
    "ref_lm_name": "QuickRead/pegasus-reddit-7e05",   # find out about the ref model
    "cls_model_name": "SophieTr/RM_incr_lr_v1",   # reward model
    "tk_name": "QuickRead/pegasus-reddit-7e05",    # tokenizer name
    "steps": 25600,
    "batch_size": 8,
    "forward_batch_size":1,
    "ppo_epochs": 1,   
    "txt_in_len": 5,
    "txt_out_len": 15,
    "lr": 1.41e-5,          # check this in the paper
    "init_kl_coef":0.2,     # check this in the paper
    "target": 6,
    "horizon":10000,
    "gamma":1,              # also check these in the paper
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

RUN_NAME = "PP0_rm_v1"
PATH = "./" + RUN_NAME
CHECKPOINT_PATH = os.path.join(PATH, 'epoch-16.pth') #'latest_minibatch.pth')

supervised_baseline = PegasusForConditionalGeneration.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")
tokenizer = PegasusTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05", cache_dir="HF_HOME")

# Policy model
policy = PegasusWithValueHead(supervised_baseline)
policy_ref = PegasusWithValueHead(supervised_baseline)

ppo_trainer = PPOTrainer(policy, policy_ref, **config)
ppo_trainer.load_state_dict(torch.load(os.path.join(RM_PATH)), strict=False)

# Data
# dataset = load_from_disk("../../../QuickRead/reddit_clean")
# train_texts, train_labels = dataset['train']['content'], dataset['train']['summary']
test_data = """
    Rain, wind and a bone-chilling cold confronted all the runners of Monday’s Boston Marathon.

    But after 20 grueling hours on the course, Maickel Melamed overcame another obstacle - a rare form of muscular dystrophy that makes it hard for him to just walk or move - to finally cross the finish line.

    Melamed, 39, may have come in last place in the 26.2-mile race, but his story touched a city now famous for its own iron-clad will after the Boston Marathon bombings in 2013.

    Scroll down for video

    Maickel Melamed, 39, crossed the Boston Marathon finish line after 20 grueling hours in the rain and cold

    Melamed has a rare form of muscular dystrophy that makes it hard just to move or walk, but he didn’t let that stop him from finishing his fifth marathon at 5am Tuesday morning

    ’After 20 hours of rain, wind and cold, Boston is still strong,’ the Venezuelan athlete said on Tuesday as he was honored at City Hall.

    ’The whole city has been so helpful and loving. The message here is that love is so much stronger than death. It was an honor to run the streets of this city.’

    Melamed walked the race with Vamos, a volunteer team from Caracas, as well as his physical trainers and dozens of friends and supporters who were there to watch him finish the race at 5am Tuesday morning.

    The athlete, who has completed four other marathons, said Boston’s hilly track became especially tough around mile 24, but his physical trainers found a way to keep him going.

    ’I’d rest 10 seconds, then take four to six steps,’ Melamed said. ‘It was a real exciting way to finish.’

    When Melamed needed to rest, collapsing into his group’s arms, they would push him back up and count his every step, according to CBS Boston.

    The athlete, who has completed four other marathons, said Boston’s hilly track became especially tough around mile 24, but his physical trainers found a way to keep him going

    Boston Mayor Marty Walsh presented a medal to Melamed and called his story ‘truly one of inspiration’

    And as the inspirational athlete inched closer and closer to the finish line, his supporters were cheering him on and yelling ‘Si se puede, si se puede!’, which is Spanish for ‘yes, we can!’

    It was Melamed’s desire to prove to others that they could achieve their dreams that kept him going.

    ’You have to know why you’re doing it, because in the last mile, the marathon will ask you if you have a reason, and if you don’t have it, you will quit,’ he told MassLive.com.

    ’Raise the bar of your own expectations for yourself. Human power is infinite.’

    There was a special reason why Melamed, who has completed races in Chicago, New York, Berlin and Tokyo, decided Boston would be the location of his last marathon.

    It was at Boston Children’s Hospital that Melamed, who was only given seven days to live when he was born, had a life-saving operation.

    Now the athlete, who has also parachuted, paraglided and climbed Venezuela’s highest mountain, must retire his running shoes because of the races’ physical toll.

    But his story, which Boston Mayor Marty Walsh called ‘truly one of inspiration,’ will continue racing on.
"""
query = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt').input_ids
response = policy.generate(query) # will not produce text
resp_txt = tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'RESPONSE from test_data is:\n {resp_txt}')