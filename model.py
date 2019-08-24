#!/usr/bin/env python3
import collections
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import numpy as np
import pickle

def load_data():    
    with open('./data/data.txt','r') as file:
        raw_dataset = file.readlines() # we map every poems into a "sentence"
    return raw_dataset


"sub-sampling the dataset to remove frequenly appear words"
def discard(idx, counter, idx_to_token, num_tokens):
    prob_discard = 1 - math.sqrt(1e-4 / (counter[idx_to_token[idx]]/num_tokens))
    return random.uniform(0, 1) <= prob_discard

"Prepare the dataset"
def create_idx_dataset(raw_dataset, threshold):
    """
    raw_dataset: list contains all poems
    threshold: frequency below this number will be discarded
    
    """
    # Count the words   
    counter = collections.Counter([tk for st in raw_dataset for tk in st])
    counter = dict(filter(lambda x: x[1] >= 10, counter.items())) # only care about words that have frequency >=10
    
    # remove special marks
    for remove_key in ['\n','□', '、', '。','《','》','「','」']: # remove marks
        try:
            del counter[remove_key]
        except:
            continue
    
    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx.keys()]
               for st in raw_dataset]
    num_tokens = sum([len(st) for st in dataset])
    subsampled_dataset = [[tk for tk in st if not discard(tk, counter, idx_to_token, num_tokens)] for st in dataset]
    print(f'Previous # of token: {num_tokens}')
    print(f'Now # of token: {sum([len(st) for st in subsampled_dataset])}')
    
    with open(f"./data/idx_to_token.pkl", "wb") as fp:
        pickle.dump(idx_to_token, fp) 
    with open(f"./data/token_to_idx.pkl", "wb") as fp:
        pickle.dump(token_to_idx, fp) 
        
    return counter, idx_to_token, token_to_idx, num_tokens, subsampled_dataset

"get centers and contexts"          
def get_centers_and_contexts(dataset, max_window_size):
    if max_window_size < 2:
        raise ValueError("The max length of window should be atleast 2")
        
    centers = [tk for st in dataset for tk in st if len(st)>=2]
    contexts = []
    for st in dataset:
        if len(st) < 2:
            continue
        for center in range(len(st)):
            window_size = random.randint(1, max_window_size)
            start_idx = max(0, center - window_size)
            end_idx = min(len(st), center + window_size + 1)
            indices = list(range(start_idx, end_idx))
            indices.remove(center)
            contexts.append(np.array(st)[np.array(indices)].tolist())
    return centers, contexts

"apply negative sampling"
def get_negatives(counter, num_tokens, all_centers, all_contexts, weight, idx_to_token, K):
    all_negatives = []
    sampling_weights = [counter[w]**weight/num_tokens for w in idx_to_token]
    population = list(range(len(idx_to_token)))
    neg_candidates = random.choices(
                    population, sampling_weights, k=int(1e6))
    idx = 0 # trace how many white noices have been used
    idx_context = 0 # trace which context we are working on
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if idx == len(neg_candidates): # if the previous generate white noices are used up
                idx = 0
                neg_candidates = random.choices(population, sampling_weights, k=int(1e6))
            neg = neg_candidates[idx]
            idx = idx + 1
            "negative words cannot be canter words or contexts word"
            if neg not in set(contexts) and neg!=all_centers[idx_context]:
                negatives.append(neg)
        all_negatives.append(negatives)
        idx_context = idx_context + 1
    return all_negatives    

"make sure center, context, negatives have the same length"
def check_lenghth(all_centers, all_contexts, all_negatives):
    assert len(all_centers) == len(all_contexts)
    assert len(all_contexts) == len(all_negatives)

"create get batch function"
def get_batch(data):
    max_len = np.array([len(context)+len(negative) for center, context, negative in data]).max()
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        total_len = len(context) + len(negative)
        centers = centers + [center]
        "fill the extra space with -1"
        contexts_negatives = contexts_negatives + [context + negative + [-1] * (max_len - total_len)] 
        "masks record the position of filled -1 and actual words"
        masks = masks + [[1] * total_len + [0]*(max_len - total_len)]
        labels = labels + [[1] * len(context) + [0]*(max_len - len(context))]
    
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))

"create data_iter function"
def create_data_iter(get_batch, batch_size, all_centers, all_contexts, all_negatives):
    num_workers = 0 if sys.platform.startswith('win32') else 4
    dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
                                 batchify_fn=get_batch, num_workers=num_workers)
    
    for batch in data_iter:
        for name, data in zip(['centers', 'contexts_negatives', 'masks',
                               'labels'], batch):
            print(name, 'shape:', data.shape)
        break
    
    return data_iter

"define the skip_gram model"
def skip_gram_model(center, contexts_and_negatives, embed_v, embed_u):
    """
    center: (batch_size, 1)
    contexts_and_negatives: (batch_size, max_len)
    
    """
    v = embed_v(center) # center
    u = embed_u(contexts_and_negatives) # context
    """
    v: (batch_size, 1, embed_size)
    u: (batch_size, max_len, embed_size)
    
    """
    
    
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    """
    pred: (batch_size, 1, max_len)
    
    """
    return pred

"train the model"
def train(embed_size, idx_to_token, lr, num_epochs, ctx, data_iter, batch_size):
    net = nn.Sequential()
    net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
            nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))
    
    net.initialize(ctx=ctx, force_reinit=True)
    loss_fun = gloss.SigmoidBinaryCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    
    for epoch in range(num_epochs):
        start_time = time.time()
        loss_sum = 0.0
        n = 0

        for batch in data_iter:
            center, context_negative, mask, label = (data.as_in_context(ctx) for data in batch)
            with autograd.record():
                pred = skip_gram_model(center, context_negative, net[0], net[1])
                l = (loss_fun(pred.reshape(label.shape), label, mask) * mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size) # we haven't divide loss by batch size
            loss_sum  =  loss_sum + l.sum().asscalar()
            n = n + l.size
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, loss_sum / n, time.time() - start_time))
        
        net.save_parameters(f'./data/params_{epoch+1}')
