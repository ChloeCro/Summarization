import re
import nltk
import string
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial

def clean_tokenized(tokenized_text):
    # clean the tokenized text 
    clean = [re.sub(r'[^\w\s]','',sentence.lower()).replace('\n', '') for sentence in tokenized_text]
    
    return clean

def remove_stopwords(tokenized_text):
    # import stopwords from NLTK
    stop_words = stopwords.words('dutch')

    # remove the stopwords and keep remaining tokens
    sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in tokenized_text]

    return sentence_tokens

def tokenize_sent(text):
    # tokenize
    sentences = sent_tokenize(text)

    return sentences

def tokenize_word(text):
    tokens = word_tokenize(text)
    return tokens

def chunk_text(text):
    pass

def get_embeddings(tokens):

    w2v=Word2Vec(tokens,vector_size=1,min_count=1,epochs=1000)
    # Initialize an empty list to store sentence embeddings
    sentence_embeddings = []

    # Calculate the sentence embeddings
    for words in tokens:
        word_embeddings = [w2v.wv[word] for word in words if word in w2v.wv]
        if word_embeddings:
            # If there are word embeddings for the words in the sentence
            sentence_embedding = np.mean(word_embeddings, axis=0)
            sentence_embeddings.append(sentence_embedding)
        #else:
            # Handle the case where no word embeddings are found for the sentence
            # You can choose to skip or assign a default value here
        #    sentence_embeddings.append(np.zeros(w2v.vector_size))

    
    #sentence_embeddings=[[w2v[word][0] for word in words] for words in tokens]
    #max_len=max([len(tokens) for tokens in tokens])
    #sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]
    
    return sentence_embeddings

def get_similarity_matrix(tokens, embeddings):
    similarity_matrix = np.zeros([len(tokens), len(tokens)])
    for i,row_embedding in enumerate(embeddings):
        for j,column_embedding in enumerate(embeddings):
            similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
    
    return similarity_matrix