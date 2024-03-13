from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

base_model = "/data/jiangmeng/test/Pretrain_model/hugging_face_LLAMA_weights_7B"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.eval()
f = open('/data/jiangmeng/ml-1m/movies.dat', 'r', encoding='ISO-8859-1')
lines = f.readlines()
f.close()
text = [_.split('::')[1].strip(" ") for _ in lines]
tokenizer.padding_side = "left"
# lines[0]
# text[:5]

from tqdm import tqdm
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

item_embedding = []
for i, batch_input in tqdm(enumerate(batch(text, 16))):
    input = tokenizer(batch_input, return_tensors="pt", padding=True)
    input_ids = input.input_ids
    attention_mask = input.attention_mask
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    item_embedding.append(hidden_states[-1][:, -1, :].detach().cpu())
    # break

item_embedding = torch.cat(item_embedding, dim=0)
torch.save(item_embedding, '/data/jiangmeng/ml-1m-split/moive_embedding.pt')