'''
input:
	data/raw_data.csv

output:
	data/sentence2embedding.pkl (preprocessing)
	protocol_embedding
'''

import csv
import pickle
from functools import reduce
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline
import openai
import os
from dotenv import load_dotenv
import tiktoken
from openai import OpenAIError
import time

torch.manual_seed(0)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key")

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://cmu.litellm.ai"
)

MODEL_NAME = "text-embedding-3-large"
TOKEN_LIMIT = 8192
CHUNK_SIZE = 2048

encoding = tiktoken.encoding_for_model(MODEL_NAME)

def get_all_protocols() -> list[str]:
    input_file = 'data/new_data.csv'
    with open(input_file, 'r') as csvfile:
        data = pd.read_csv(input_file)
    protocols = dict()
    for index, row in data.iterrows():
        nctid = row[0]
        protocol = row[1]
        protocols[nctid] = protocol
    return protocols

def collect_cleaned_sentence_set():
	protocol_lst = get_all_protocols() 
	cleaned_sentence_lst = dict()
	for icd in protocol_lst.keys():
		protocol_clean = protocol_lst[icd].strip().lower()
		cleaned_sentence_lst[icd] = protocol_clean
	return cleaned_sentence_lst

# def save_sentence_gpt_dict_pkl():
#     cleaned_sentence_set = collect_cleaned_sentence_set()
#     def text2vec(text):
#         return client.embeddings.create(
# 			model = "text-embedding-3-large",
# 			input = text)
#     protocol_sentence_2_embedding = dict()
#     for icd in tqdm(cleaned_sentence_set.keys()):
#         embed = text2vec(cleaned_sentence_set[icd])
#         embeddings = [torch.tensor(e.embedding) for e in embed.data]
#         a = torch.stack(embeddings)
#         protocol_sentence_2_embedding[icd] = a
#         #
#     pickle.dump(protocol_sentence_2_embedding, open('data/icd2embedding.pkl', 'wb'))

def truncate_or_split(text: str, max_tokens: int = TOKEN_LIMIT, chunk_size: int = CHUNK_SIZE) -> list[str]:
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    else:
        return [encoding.decode(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

def embed_chunks(chunks: list[str], retries: int = 5) -> torch.Tensor:
    for attempt in range(retries):
        try:
            response = client.embeddings.create(input=chunks, model=MODEL_NAME)
            embeddings = [torch.tensor(e.embedding) for e in response.data]
            return torch.stack(embeddings).mean(dim=0)  # mean pooling
        except OpenAIError as e:
            print(f"[Retry {attempt+1}/{retries}] Error: {e}")
            time.sleep(2 ** attempt + 1)
    raise RuntimeError("Max retries reached while embedding.")

def batch_embed_texts(texts: list[str], max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=MODEL_NAME,
                input=texts
            )
            return [torch.tensor(e.embedding) for e in response.data]
        except OpenAIError as e:
            print(f"[Retry {attempt+1}] Failed batch: {e}")
            time.sleep(2 ** attempt + 1)
    raise RuntimeError("Max retries exceeded for batch.")

def save_sentence_gpt_dict_pkl(batch_size = 4):
    cleaned_sentence_set = collect_cleaned_sentence_set()
    protocol_sentence_2_embedding = {}

    output_path = 'data/icd2embedding.pkl'
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            protocol_sentence_2_embedding = pickle.load(f)
        print(f"Loaded {len(protocol_sentence_2_embedding)} previously saved embeddings.")

    items = [(k, v) for k, v in cleaned_sentence_set.items() if k not in protocol_sentence_2_embedding.keys()]
    for i in tqdm(range(0, len(items), batch_size)):
        batch = items[i:i+batch_size]
        keys = [k for k, v in batch]
        texts = [v for k, v in batch]

        long_texts = [t for t in texts if len(encoding.encode(t)) > TOKEN_LIMIT]
        if long_texts:
            for k, t in zip(keys, texts):
                tokens = encoding.encode(t)
                if len(tokens) > TOKEN_LIMIT:
                    chunks = truncate_or_split(t)
                    try:
                        emb = embed_chunks(chunks)
                        protocol_sentence_2_embedding[k] = emb
                    except Exception as e:
                        print(f"Failed long input {k}: {e}")
            continue

        try:
            embeddings = batch_embed_texts(texts)
            for k, emb in zip(keys, embeddings):
                protocol_sentence_2_embedding[k] = emb
        except Exception as e:
            print(f"Failed batch starting at {i}: {e}")
            continue

        if len(protocol_sentence_2_embedding) % 100 == 0:
            with open(output_path, 'wb') as f:
                pickle.dump(protocol_sentence_2_embedding, f)

    with open(output_path, 'wb') as f:
        pickle.dump(protocol_sentence_2_embedding, f)

def load_sentence_2_vec():
	sentence_2_vec = pickle.load(open('data/icd2embedding.pkl', 'rb'))
	return sentence_2_vec
	
class Protocol_Embedding(nn.Sequential):
    def __init__(self, output_dim, highway_num, device):
        super(Protocol_Embedding, self).__init__()
        self.input_dim = 3072
        self.output_dim = output_dim
        self.highway_num = highway_num
        self.fc = nn.Linear(self.input_dim, output_dim)
        self.f = F.relu #gelu
        self.device = device
        self = self.to(device)

    def forward_single(self, feature):
        # inclusion_feature, exclusion_feature: xxx,3072
        feature = feature.to(self.device)
        feature_vec = torch.mean(feature, 0)
        feature_vec = feature_vec.view(1, -1)
        return feature_vec

    def forward(self, feature):
        result = [self.forward_single(f_mat) for f_mat in feature] 
        result_mat = torch.cat(result, 0)  # 32,3072
        output = self.f(self.fc(result_mat))
        return output
    
    @property
    def embedding_size(self):
        return self.output_dim
	
if __name__ == "__main__":
	save_sentence_gpt_dict_pkl()