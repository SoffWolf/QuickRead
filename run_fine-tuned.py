import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusForConditionalGeneration, PegasusTokenizer
from pathlib import Path  

from PPO_training.pegasus_with_heads import PegasusWithValueHead
# My fine-tuned
# tokenizer = AutoTokenizer.from_pretrained("SophieTr/fine-tune-Pegasus")
# model = AutoModelForSeq2SeqLM.from_pretrained("SophieTr/fine-tune-Pegasus")
 
# # Original
# model_origin = PegasusForConditionalGeneration.from_pretrained('sshleifer/distill-pegasus-xsum-16-4')
# tokenizer_origin = PegasusTokenizer.from_pretrained('sshleifer/distill-pegasus-xsum-16-4')

# inp = "In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs."
# input_ids = tokenizer(inp, return_tensors="pt").input_ids
# outputs = model.generate(input_ids=input_ids)
# print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
# print("length: ", output.shape, "   ", input_ids.shape)

# input_ids_benchmark = tokenizer_origin(inp, return_tensors="pt").input_ids
# outputs_benchmark = model_origin.generate(input_ids=input_ids_benchmark)
# print("Generated from benchmark\n:", tokenizer.batch_decode(outputs_benchmark, skip_special_tokens=True))
print(len(sys.argv))
if len(sys.argv) != 3:
    raise ValueError('Please provide:\n(1) the input post txt file, and\n(2) model name: "QuickRead/pegasus-reddit-7e05", "PPO_v8", "PPO_v9".')

def preprocess(inp):
    input_ids = tokenizer(inp, return_tensors="pt").input_ids
    return input_ids
def predict(input_ids):
    outputs = model.generate(input_ids=input_ids)
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)#[0]
    return res

if __name__ == '__main__':
    file_name = sys.argv[1]
    model_name = sys.argv[2]
    print(f'file name = "{file_name}", and model_name = "{model_name}"')
    txt = ""
    with open(file_name,'r') as line: 
        txt += line.read()
    print("The original post is:\n", txt)
    supervised_baseline_name = "QuickRead/pegasus-reddit-7e05"
    tokenizer = AutoTokenizer.from_pretrained(supervised_baseline_name, cache_dir="HF_HOME")
    try: 
        if model_name == "QuickRead/pegasus-reddit-7e05":
            model = AutoModelForSeq2SeqLM.from_pretrained("QuickRead/pegasus-reddit-7e05")

        else:
            PATH = "./PPO_training/" + model_name
            CHECKPOINT_PATH = os.path.join(PATH, 'latest_minibatch.pth')
            supervised_baseline = PegasusForConditionalGeneration.from_pretrained(supervised_baseline_name, cache_dir="HF_HOME")
            policy = PegasusWithValueHead(supervised_baseline)
            policy.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH), map_location=torch.device('cpu')), strict=False)
        inp_ids = preprocess(txt)
        print('Model generated Summary:\n', predict(inp_ids))
    except Exception as e:
        print("Error occurs when trying to generate summary: ", e)
        print('"Check model_name arg in: "QuickRead/pegasus-reddit-7e05", "PPO_v8", "PPO_v10"')