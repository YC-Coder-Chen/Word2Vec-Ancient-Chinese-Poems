#!/usr/bin/env python3
from mxnet.gluon import nn
import pickle

with open(f"./data/idx_to_token.pkl", "rb") as fp:
    idx_to_token = pickle.load(fp) 
with open(f"./data/token_to_idx.pkl", "rb") as fp:
    token_to_idx = pickle.load(fp) 

# User define, has to be in consistance with train_model
embed_size = 150
directory = './data/params_25'
query_token = '美'

net = nn.Sequential()
net.add(nn.Embedding(input_dim = len(idx_to_token), output_dim = embed_size),
        nn.Embedding(input_dim = len(idx_to_token), output_dim = embed_size))
net.load_parameters(directory)

def get_similar_tokens(query_token):
    W = net[0].weight.data()
    try:
        vector = W[token_to_idx[query_token]] 
        return vector
    except:
        raise ValueError("The word is not included in the model")

get_similar_tokens(query_token)