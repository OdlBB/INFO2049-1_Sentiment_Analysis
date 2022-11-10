# Git repository

https://github.com/OdlBB/INFO2049-1_Sentiment_Analysis

# Summary
This repository contains two experiment notebooks based on two different datasets : The IMDB reviews dataset (```IMDB SentimentAnalysis.ipynb```) and the Tweets dataset (```Twitter SentimentAnalysis.ipynb```). The goal of the experiments is to classify the sentiment of the reviews and tweets as positive or negative. 


These notebooks contain various models for the sentiment analysis of both datasets. The datasets can be downloaded at the following url:

- IMDB reviews dataset : https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/data

- Tweets dataset : https://www.kaggle.com/code/paoloripamonti/twitter-sentiment-analysis


For this project, we have the choice between several embeddings:
- Word emmbeddings:
    - Pre-trained Glove (glove.42B.300d.txt, https://www.kaggle.com/datasets/yutanakamura/glove42b300dtxt)
    - Pre-trained Word2Vec (GoogleNews-vectors-negative300.bin, https://developer.syn.co.in/tutorial/bot/oscova/pretrained-vectors.html)
    - Word2Vec trained from our training set
    - Pre-trained FastText (wiki-news-300d-1M.vec, https://fasttext.cc/docs/en/english-vectors.html)
- Document embedding:
    - Doc2Vec
    - Word2Vec averaged with TF-IDF

We also have the possibility to choose between several RNN:
- LSTM with attention
- GRU with attention
- LSTM without attention
- GRU without attention


# Build the datasets

For the Twitter dataset : Initially the dataset was constituted of two separate ```.csv``` files with a training set, and a testing set, so no operation was needed to split the dataset.

For the IMDB dataset:
Initially, we only had one dataset. However, for the project, we need a train and test set. For that, we first load the whole set, separate it and both of them
into ```.csv``` files (train_imbd and test_imbd). This needs to only be done. Once we
have the  ```.csv ```files, we can directly jump to the pre-processing step.

# Pre-processing

In this step, we can load the train and test sets and pre-process them (more
details in the notebook and the report on what is done for the pre-processing).
Each document in both sets is now a list of tokens.

# Build vocabulary
🚨 If you want to use **document embedding**, this step can be skipped. 🚨

From the list of tokens available in the training set, we build a vocabulary 
that will later be used to build the embedding matrix. From this vocabulary,
we build 2 dictionaries :
- word2index: for a given word, returns the id number
- index2word: for a given id, returns the corresponding word

# Build word embedding
🚨 If you want to use **document embedding**, this step can be skipped. 🚨

In order to choose which word_embedding to use, there is the variable 
```EMBEDDING_TYPE``` which is a string and can have the following values:
- ```word2vec```
- ```glove```
- ```fasttext```

If Word2Vec was chosen, there are 2 other values to initalize
- ```PRE_TRAINED```
    - ```True``` to use the pre-trained embedding.
    - ```False``` otherwise.
- ```RETRAIN_EMBEDDINGS```
    - ```True``` to train a new embedding.
    - ```False``` to load a previous trained word embedding.

# Padding sequences and dataloader
🚨 If you want to use **document embedding**, this step can be skipped. 🚨

The input of the model has constant/fixed size. Thus, all sequencesmust be the same size, which is achieved thanks to the 
token ```<PAD>```. Finally, all that remains is to put the data into the right type and separate it into batches.

# Document embedding
🚨 If you want to use **word embedding**, this step can be skipped. 🚨

In this section, we build the document embeddings and we return that data
directly embedded for the training (no embedding layer in RNN). We can chose
which document embedding layer to use with the variable document_model which
can take the following values:
- ```doc2vec```
- ```word2vec-tfidf```

# Models

In this section, several models are built. We have : 

- RNN (```RNN```)
- Transformer using only attention (unreported, for information purposes.)
- RNN with attention layer (```AttentiveRNN```)
- RNN with attention for document embedding (```AttentiveRNN_Doc_emb```)
- RNN for document embedding (```RNN_Doc_emb```)

# Training

In order to select the model to use, we need to use the variable ```MODEL_NAME```
which can have the following values:
- ```RNN```
- ```AttentiveRNN```
- ```Transformer```
- ```DocumentEmbs```

We also need to choose the value of the ```RNN_TYPE``` variable which can be:
- ```GRU```
- ```LSTM```


# Load the model

This section allows to re-load some previously trained models in order to test them for inference, for instance, instead of re-training a new model every time.

# Evaluate the model

Finally, we can evaluate our model thanks to our test set. We obtain the 
following performance metrics:
- Confusion matrix
- Accuracy
- Precision
- Recall

We also built a predict method that predicts if a given sentence is positive or 
negative. However, it isn't totally reliable as these sentence do not undergo
the pre-processing process.
