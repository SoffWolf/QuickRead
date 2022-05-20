# Calculate multiple rouge score of the summarization models: supervised baseline, final policy model, other summarization model, for ex: PEGASUS xsum
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from pathlib import Path  

SAVEPATH = Path('out.csv') 

DATAPATH = 'data/human_feedback.parquet'
df = pd.read_parquet(DATAPATH, engine="pyarrow")

df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df)), int(.95*len(df))])

input_posts = df_test[:10]['post']
label_summaries_1 = df_test[:10]['summary1']
label_summaries_2 = df_test[:10]['summary2']

model_name = "QuickRead/pegasus-reddit-7e05"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

## Model generates: 
def preprocess(inp):
    input_ids = tokenizer(inp, return_tensors="pt", truncation=True).input_ids
    return input_ids
def predict(input_ids):
    outputs = model.generate(input_ids=input_ids)
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return res

def get_embedding(sentences):
    s_bert = SentenceTransformer('all-distilroberta-v1')
    # print(model)
    print("------------------------------->_<-------------------------------")
    embeddings = s_bert.encode(sentences)
    return embeddings

# Model generate
# TODO: Add logic: if file model_generated.pyarrow not existed: run the prediction and save result to file --> else: load prediction
out = []
for post in input_posts:
  tokens = preprocess(post)
  response = predict(tokens)
  out.append(response)

# calculate the scores here
# calculate rouge
rouge_1_lst = []
rouge_2_lst = []
sm1_lst = []
sm2_lst = []
df = pd.DataFrame(columns=[ '1-rouge-1f', '1-rouge-2f', '1-rouge-lf', '2-rouge-1f', '2-rouge-2f', '2-rouge-lf', '1-cosine-sim', '2-cosine-sim'])
for i in range(0, len(out)+1, 10):
    rouge = Rouge()
    rouge_score_1 = rouge.get_scores(out[i:i+10], label_summaries_1[i:i+10], avg=True)
    rouge_score_2 = rouge.get_scores(out[i:i+10], label_summaries_2[i:i+10], avg=True)

    #calculate sbert
    d_out = get_embedding(out[i:i+10])

    d_1 = get_embedding(label_summaries_1[i:i+10])
    sm1_mat = util.cos_sim(d_1, d_out)
    sm1 = sum([sm1_mat[k][k] for k in range(10)])/10


    d_2 = get_embedding(label_summaries_2[i:i+10])
    sm2_mat = util.cos_sim(d_2, d_out)
    sm2 = sum([sm2_mat[k][k] for k in range(10)])/10

    # append that shit
    rouge_1_lst.append(rouge_score_1)
    rouge_2_lst.append(rouge_score_2)
    sm1_lst.append(sm1)
    sm2_lst.append(sm2)

    df2 = pd.DataFrame({ '1-rouge-1f': rouge_score_1['rouge-1']['f'],
                        '1-rouge-2f': rouge_score_1['rouge-2']['f'],
                        '1-rouge-lf': rouge_score_1['rouge-l']['f'],
                        '2-rouge-1f': rouge_score_2['rouge-1']['f'],
                        '2-rouge-2f': rouge_score_2['rouge-2']['f'],
                        '2-rouge-lf': rouge_score_2['rouge-l']['f'],
                        '1-cosine-sim': sm1,
                        '2-cosine-sim': sm2})
    df = df.append(df2, ignore_index = True)


for column in df:
    print(column.mean())

df.to_csv(SAVEPATH,index=False)


