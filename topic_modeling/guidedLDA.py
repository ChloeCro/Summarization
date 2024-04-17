import pandas as pd
import os
import advertools as adv
import re
import gensim.corpora as corpora

from gensim.models.coherencemodel import CoherenceModel
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.feature_extraction.text import CountVectorizer


df=pd.read_csv("data/fulltext_by_year/2022_rs_data.csv")

stopwords_dutch = list(adv.stopwords['dutch'])
print(stopwords_dutch)

tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-dutch")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')#"GroNLP/gpt2-small-dutch")

vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords_dutch)

docs = df.fulltext.tolist()
docs = [str(doc) if not pd.isna(doc) else "" for doc in docs]

embeddings = embedding_model.encode(docs)

nr_topics = [5, 10, 15]

procesverloop = ['procesverloop', 'besluit']
mat_feiten = ['feiten']
prev_beslissingen = ['gerechtshof', 'rechtbank', 'besluit']
middelen = ['middelen']
verweer = ['verweer']
beoordeling = ['beoordeling', 'beoordeelt']
conclusie = ['verklaart', 'gegrond', 'ongegrond']

seed_topic_list=[procesverloop,mat_feiten,prev_beslissingen,middelen, verweer, beoordeling, conclusie]
print(seed_topic_list)

coherence_list = []
best_model = None
best_coherence = 0
timestamps = df.date.to_list()
models = []

for topic in nr_topics:
    BERTopic_model = BERTopic(embedding_model = embedding_model, vectorizer_model= vectorizer_model, language='dutch', calculate_probabilities=True, seed_topic_list=seed_topic_list, nr_topics=topic)
    topics, _ = BERTopic_model.fit_transform(docs, embeddings)
    topics_over_time = BERTopic_model.topics_over_time(docs, timestamps)
    
    # Preprocess documents
    cleaned_docs = BERTopic_model._preprocess_text(docs)

    # Extract vectorizer and tokenizer from BERTopic
    vectorizer = BERTopic_model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [tokenizer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in BERTopic_model.get_topic(topic)] 
               for topic in range(len(set(topics))-1)]

    # Evaluate (OCTIS?)
    coherence_model = CoherenceModel(topics=topic_words, 
                                 texts=tokens, 
                                 corpus=corpus,
                                 dictionary=dictionary, 
                                 coherence='c_v')
    coherence = coherence_model.get_coherence()
    coherence_list.append(coherence)
    
    # Save best model
    if best_coherence < coherence:
        best_model = BERTopic_model
        best_coherence = coherence
    
    # Save all models
    models.append(BERTopic_model)