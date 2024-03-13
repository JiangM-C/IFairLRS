from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import os
import math
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
parse = argparse.ArgumentParser()

args = parse.parse_args()

import pandas as pd
data = pd.read_csv("/data/ml-1m-split/test/test.csv")
index = data.index

path = []
path.append(os.path.join('/data/test/result', "ml-1m_result.json"))
print(path)

# 读取模型
base_model = "/data/test/Pretrain_model/hugging_face_LLAMA_weights_7B"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

f = open('/data/ml-1m/ratings.dat', 'r')
data = f.readlines()
f.close()


f = open('/data/ml-1m/movies.dat', 'r', encoding='ISO-8859-1')
movies = f.readlines()
movie_names = [_.split('::')[1].strip("\"") for _ in movies]
movie_ids = [_.split('::')[0] for _ in movies]
movie_dict = dict(zip(movie_names, movie_ids))
id_mapping = dict(zip(movie_ids, range(len(movie_ids))))
f.close()

movie_genre = pd.read_csv("/data/ml-1m-split/movies_genre.csv")
movie_genre = movie_genre.drop(columns=["Title"])
genre_set = movie_genre.columns.to_list()

# 统计test set中的不同类别的比例
test_data = pd.read_csv("/data/ml-1m-split/test/test.csv")
history_list = test_data["history_movie_id"].to_list()
history_list = [eval(_) for _ in history_list]
history_count = {_:0 for _ in genre_set}

for l in history_list:
    for id in l:
        index = id_mapping[id]
        
        movie = movie_genre.iloc[index]
        for col in movie.index:
            history_count[col] += int(movie[col])
        

result_dict = dict()
for p in path:

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    f = open(p, 'r')
    import json
    test_data = json.load(f)
    f.close()

    text = [_["predict"][0].strip("\"") for _ in test_data]
    tokenizer.padding_side = "left"

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]

    predict_embeddings = []

    from tqdm import tqdm
    for i, batch_input in tqdm(enumerate(batch(text, 16))):
        input = tokenizer(batch_input, return_tensors="pt", padding=True)
        input_ids = input.input_ids
        attention_mask = input.attention_mask
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())

    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    # if p.find("des") == -1:
    movie_embedding = torch.load("/data/ml-1m-split/moive_embedding.pt").cuda()

    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)

    for gamma in [0]:

        # topk_list = [1, 3, 5, 10, 20, 50]

        # rank = torch.pow((1 + pop_rank), -gamma) * dist
        # # print(rank)
        rank = dist.argsort(dim = -1) # .argsort(dim = -1)

        topk_list = [1, 3, 5, 10, 20, 50]
        topk_count = {k: {_:0 for _ in genre_set} for k in topk_list}

        for topk in topk_list:
            for i in range(len(test_data)):
                for k in range(topk):
                    movie = movie_genre.iloc[rank[i][k].item()]
                    for col in movie.index:
                        topk_count[topk][col] += int(movie[col])


        f = open('./log/movie_all_fairness_on_ml_1m.json', 'w') 
        history_count = {key: int(value) for key, value in history_count.items()}
        for tpok in topk_list:
            topk_count[topk] = {key: int(value) for key, value in topk_count[topk].items()}
        json.dump([history_count, topk_count], f, indent=4)

        json.dump(topk_count, f, indent=4)
        f.close()

     