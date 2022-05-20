# Calculate multiple rouge score of the summarization models: supervised baseline, final policy model, other summarization model, for ex: PEGASUS xsum
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sentence_transformers import util



df = pd.read_parquet(DATAPATH, engine="pyarrow")

df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df)), int(.95*len(df))])

input_posts = df_test[:10]['post']
label_summaries_1 = df_test[:10]['summary1']
label_summaries_2 = df_test[:10]['summary2']

model_name = "QuickRead/pegasus-reddit-7e05"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(inp):
    input_ids = tokenizer(inp, return_tensors="pt", truncation=True).input_ids
    return input_ids
def predict(input_ids):
    outputs = model.generate(input_ids=input_ids)
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return res
def get_embedding(sentences):
    model = SentenceTransformer(model_name)
    print(model)
    print("------------------------------->_<-------------------------------")
    embeddings = model.encode(sentences)
    return embeddings

# calculate the scores here
out = []
for post in input_posts:
  tokens = preprocess(post)
  response = predict(tokens)
  out.append(response)

# calculate rouge
rouge_1_lst = []
rouge_2_lst = []
sm1_lst = []
sm2_lst = []

for i in range(0, len(out)-10, 10):

    rouge = Rouge()
    rouge_score_1 = rouge.get_scores(out[i:i+10], label_summaries_1[i:i+10], avg=True)
    rouge_score_2 = rouge.get_scores(out, label_summaries_2, avg=True)
    #calculate sbert
    d_out = get_embedding(out[i:i+10])

    d_1 = get_embedding(label_summaries_1[i:i+10])
    sm1_mat = util.cos_sim(d_1, d_out) # Runtime = 2s
    sm1 = sum([sm1_mat[k][k] for k in range(10)])/10


    d_2 = get_embedding(label_summaries_2[i:i+10])
    sm2_mat = util.cos_sim(d_2, d_out) # Runtime = 2s
    sm2 = sum([sm2_mat[k][k] for k in range(10)])/10

    # append that shit
    rouge_1_lst.append(rouge_score_1)
    rouge_2_lst.append(rouge_score_2)
    sm1_lst.append(sm1)
    sm2_lst.append(sm2)


