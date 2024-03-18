# TextRank: https://medium.com/data-science-in-your-pocket/text-summarization-using-textrank-in-nlp-4bce52c5b390
# LexRank
#

import numpy as np
import pandas as pd
import networkx as nx

from summ_pipeline.utils.preprocess_text import tokenize_sent, clean_tokenized, remove_stopwords, get_embeddings, get_similarity_matrix

def textRank(text, n_sent):
    # clean and tokenize text
    tokenized = tokenize_sent(text)
    clean_tokens = clean_tokenized(tokenized)
    no_stopwords_tokens = remove_stopwords(clean_tokens)

    # get the sentence embeddings
    embeddings = get_embeddings(no_stopwords_tokens)

    # get the similarity matrix
    similarity_matrix = get_similarity_matrix(no_stopwords_tokens, embeddings)

    # convert similarity matrix to a graph
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    top_sentence={sentence:scores[index] for index,sentence in enumerate(tokenized)}
    top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:n_sent])

    top_sentences = []
    for sent in tokenized:
        if sent in top.keys():
            top_sentences.append(sent)

    return top_sentences