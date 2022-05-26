# Calculate multiple rouge score of the summarization models: supervised baseline, final policy model, other summarization model, for ex: PEGASUS xsum
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from pathlib import Path  
from tqdm import tqdm

SAVEPATH = Path('out.csv') 

DATAPATH = '../rewards/data/human_feedback.parquet'
df = pd.read_parquet(DATAPATH, engine="pyarrow")
n_sample = 10
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df)), int(.95*len(df))])

input_posts = list(df_test['post'].values)
label_summaries_1 = list(df_test['summary1'].values)
label_summaries_2 = list(df_test['summary2'].values)

model_name = "QuickRead/pegasus-reddit-7e05"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def get_embedding(sentences):
    s_bert = SentenceTransformer('all-distilroberta-v1')
    # print(model)
    
    embeddings = s_bert.encode(sentences)
    return embeddings

# Model generate
# TODO: Add logic: if file model_generated.pyarrow not existed: run the prediction and save result to file --> else: load prediction
out = list(pd.read_parquet('../PPO_training/ppo_output/out.parquet', engine="pyarrow")['summary'])
# out = []
# for post in input_posts:
# #   print(post)
# #   print("++++"*4)
#   tokens = preprocess(post)
  
#   response = predict(tokens)
#   out.append(response)
#   print(response)
#   print("------------------------------->_<-------------------------------")

# calculate the scores here
# calculate rouge
rouge_1_lst = []
rouge_2_lst = []
sm1_lst = []
sm2_lst = []
df = pd.DataFrame(columns=[ '1-rouge-1f', '1-rouge-2f', '1-rouge-lf', '2-rouge-1f', '2-rouge-2f', '2-rouge-lf', '1-cosine-sim', '2-cosine-sim'],
                  index=range(int(np.ceil(len(out)/n_sample))))
for i in tqdm(range(0, len(out), n_sample)):
    rouge = Rouge()
    rouge_score_1 = rouge.get_scores(out[i:i+n_sample], label_summaries_1[i:i+n_sample], avg=True)
    rouge_score_2 = rouge.get_scores(out[i:i+n_sample], label_summaries_2[i:i+n_sample], avg=True)

    #calculate sbert
    d_out = get_embedding(out[i:i+n_sample])

    d_1 = get_embedding(label_summaries_1[i:i+n_sample])
    sm1_mat = util.cos_sim(d_1, d_out)
    sm1 = sum([sm1_mat[k][k] for k in range(n_sample)])/n_sample


    d_2 = get_embedding(label_summaries_2[i:i+n_sample])
    sm2_mat = util.cos_sim(d_2, d_out)
    sm2 = sum([sm2_mat[k][k] for k in range(n_sample)])/n_sample

    # append that shit
    rouge_1_lst.append(rouge_score_1)
    rouge_2_lst.append(rouge_score_2)
    sm1_lst.append(sm1)
    sm2_lst.append(sm2)
    # df.loc['Jane',:] = [23, 'London', 'F']
    df.loc[int(np.ceil(i/n_sample)),:] = [rouge_score_1['rouge-1']['f'],
                        rouge_score_1['rouge-2']['f'],
                        rouge_score_1['rouge-l']['f'],
                        rouge_score_2['rouge-1']['f'],
                        rouge_score_2['rouge-2']['f'],
                        rouge_score_2['rouge-l']['f'],
                        sm1.item(),
                        sm2.item()]
    # df = df.append(df2, ignore_index = True)
    # print(df)
# print("------------------------------->_<-------------------------------")

# print(df)
for column in list(df.columns):
    print(df[column].mean())

df.to_csv(SAVEPATH,index=False)


