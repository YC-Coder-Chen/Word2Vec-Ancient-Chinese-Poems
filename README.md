Word2Vec-Ancient-Chinese-Poems
============

In this project, a word-embedding Skip-Gram model is trained on over 60k Ancient Chinese Poems (including Song Poetry and Tang Poetry)based on MXNET and Python 3.6. This project is inspired by the interactive deep learning course [Dive into Deep Learning](https://d2l.ai/). To reduce model complexity, we applied negative sampling during the model training. Subsampling technique is also utilized in this project to drop out words that appeared extremely frequently in the text but contains little meaning.  

Highlight
------------

Before this project, there is no word-embedding model specifically for ancient Chinese poems. There are models built on ancient Chinese, but that model is not suitable for projects such as Ancient Poetry Generator. Because ancient Chinese poems tend to be slightly different and words in ancient Chinese poems tend to be more concise and sophisticated.

Data
------------

The provided dataset came from [chinese-poetry project](https://github.com/chinese-poetry/chinese-poetry), a great database contains almost all the ancient poetries in Chinese. The provided trained Skip-Gram model is based on 60k five-characters eight-lines poems including Tang poetry and Song poetry from the database. Users can change the [data_cleaning.py](/data_cleaning.py) file to create your training data.

Model training
------------

User can modify the [train_model.py](/train_model.py) and run the file to train your model. The default optimizer is "SGD", users can also change the optimizer to "Adam" or other optimizers supported by MXNET in the [model.py](/model.py). More specific parameters details are provided in the file. Below is the setting parameters for the trained Skip-Gram model.

```
batch_size = 512
max_window_size = 3
k = 5 # number of negative sampling
lr = 0.001 # learning rate
num_epoch = 25
embed_size = 150

```

Model Predict
------------

User can apply the trained model to convert Ancient Chinese Character into vector representation in the [predict.py](/predict.py) file. We also examine the cosine similarity of some word vectors to check whether the model is valid or not. 

```
# first check
Input: get_similar_tokens('貴', 10, net[0])

Output: 
cosine sim=0.388: 富
cosine sim=0.377: 戒
cosine sim=0.341: 瘠
cosine sim=0.323: 非
cosine sim=0.322: 孟
cosine sim=0.322: 侯
cosine sim=0.319: 謂
cosine sim=0.318: 吃
cosine sim=0.316: 耳
cosine sim=0.312: 羨

# second check
Input: get_similar_tokens('民', 10, net[0])

Output: 
cosine sim=0.447: 治
cosine sim=0.379: 均
cosine sim=0.363: 瘼
cosine sim=0.357: 庶
cosine sim=0.345: 燮
cosine sim=0.342: 訟
cosine sim=0.342: 氓
cosine sim=0.330: 戾
cosine sim=0.328: 仁
cosine sim=0.327: 無

```
