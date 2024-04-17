import re
import nltk
import string
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial

"""
def clean_tokenized(tokenized_text):
    # clean the tokenized text 
    clean = [re.sub(r'[^\w\s]|[\d]','',sentence.lower()).replace('\n', '') for sentence in tokenized_text]

    return clean
"""

def clean_tokenized(tokenized_text):
    # Extended cleaning to include tabs, multiple newlines, and other whitespace characters
    clean = [re.sub(r'\s+', ' ', re.sub(r'[^\w\s]|[\d]', '', sentence.lower())).strip() for sentence in tokenized_text]

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

def remove_info(text):
    pattern_file = open('general_utils\patterns.txt', 'r')
    lines = pattern_file.readlines()
    patterns = [line.strip()[2:-1].replace('\\\\', '\\') for line in lines]
    super_pattern = "|".join(patterns)

    sectioning = re.split(super_pattern, text)

    text = ' '.join(sectioning[1:])

    return text

def sliding_window(text, window_size=500):
    #default max token limit is 516 tokens therefore default window size of 500
    tokenized_text = tokenize_word(text)

    overlap = window_size // 2

    slices = []

    start_index = 0

    while start_index < len(tokenized_text):
        # If it's not the first chunk, move back 'overlap' words to include them in the current chunk for the left overlap
        if start_index > 0:
            start_index -= overlap
        
        # Select words for the current chunk
        end_index = start_index + window_size
        chunk = ' '.join(tokenized_text[start_index:end_index])
        
        # Add the chunk to our list of chunks
        slices.append(chunk)

        # Update the start index for the next chunk, ensuring the right overlap
        start_index = end_index

        # If we are at the end and there's no more room for a full chunk, break the loop
        if start_index >= len(tokenized_text) - overlap:
            break

    return slices

def get_embeddings(tokens):

    for sublist in tokens:
    # Use a list comprehension to filter out empty strings from each sublist
        sublist[:] = [item for item in sublist if item != '']

    #print(tokens)
    

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