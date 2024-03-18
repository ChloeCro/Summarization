# k-means clustering: https://medium.com/@akankshagupta371/understanding-text-summarization-using-k-means-clustering-6487d5d37255
# hierarchical clustering
# mean-shift clustering?
# LDA


import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial import distance

from summ_pipeline.utils.preprocess_text import tokenize_sent, clean_tokenized, remove_stopwords, get_embeddings

def k_means(text, n_clusters):
    # tokenize sentences and clean them
    tokenized = tokenize_sent(text)
    tokenized_clean = clean_tokenized(tokenized)
    no_stopwords_tokens = remove_stopwords (tokenized_clean)

    # Word2Vec
    sentence_embeddings = get_embeddings(no_stopwords_tokens)

    # clustering
    #n_clusters = 5
    kmeans = KMeans(n_clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(sentence_embeddings)

    summ_sentence_id = []
    for i in range(n_clusters):
        sentence_dict={}
        
        for j in range(len(y_kmeans)):
            
            if y_kmeans[j]==i:
                sentence_dict[j] =  distance.euclidean(kmeans.cluster_centers_[i],sentence_embeddings[j])
        min_distance = min(sentence_dict.values())
        summ_sentence_id.append(min(sentence_dict, key=sentence_dict.get))

    summ_sentences = []
    for i in sorted(summ_sentence_id):
        #print(tokenized[i])
        summ_sentences.append(tokenized[i])

    return summ_sentences