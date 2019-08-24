#!/usr/bin/env python3
import mxnet as mx
from model import load_data, create_idx_dataset, get_centers_and_contexts, get_negatives, get_batch, check_lenghth, create_data_iter, train

# User define
batch_size = 512
max_window_size = 3
k = 5 # number of negative sampling
lr = 0.001 # learning rate
num_epoch = 25
embed_size = 150

raw_dataset = load_data()
counter, idx_to_token, token_to_idx, num_tokens, subsampled_dataset = create_idx_dataset(raw_dataset, 0.75)
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, max_window_size)
all_negatives = get_negatives(counter, num_tokens, all_centers, all_contexts, 
                              0.75, idx_to_token, k) # negative sampling k words
check_lenghth(all_centers, all_contexts, all_negatives)
data_iter = create_data_iter(get_batch, batch_size, all_centers, all_contexts, all_negatives)
train(embed_size, idx_to_token, lr, num_epoch, mx.gpu(), data_iter, batch_size)
