import os, sys
import argparse

from collections import Counter
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentencesTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification

import advertools as adv
from bertopic import BERTopic

# Setup argparser
parser = argparse.ArgumentParser(description="Topic Modelling with Sentence based ")

parser.add_argument('--data_dir', type=str, default='topic_modeling/sectioned_data_2022.csv', help="""Data directory of the unprocessed data""")
parser.add_argument('--save_dir', type=str, default='topic_modeling/models', help="""Directory for models to be saved""")
parser.add_argument('--sectioned', action='store_true', help='use sections for embeddings')
#parser.add_argument('--cuda', action='store_true', help='use CUDA')
args = parser.parse_args()

# load data

data = pd.read_csv(args.data_dir)

if args.sectioned:
    docs = [string for row_list in data['sections'] for string in row_list]
else:
    docs = data.fulltext.tolist()

stopwords_dutch = list(adv.stopwords['dutch'])

vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords_dutch)

model = SentenceTransformer("GroNLP/gpt2-small-dutch")

topic_model = BERTopic(embedding_model=model, 
                       vectorizer_model=vectorizer_model,
                       language='multilingual', calculate_probabilities=True,
                       verbose=True)

topics, probs = topic_model.fit_transform(docs)

topic_model.save("topic_modeling/models/bertopic_models")

print(topics)

# python topic_modeling/LDA_Bertopic.py 
