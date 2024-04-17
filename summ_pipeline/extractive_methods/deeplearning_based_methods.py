# BERT/BART/GPT 
# https://utomorezadwi.medium.com/bert-extractive-summarizer-vs-word2vec-extractive-summarizer-which-one-is-better-and-faster-c6d6d172cb91
#TODO: make min_length max_length an input from the pipeline


import os
import re
import sys
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import torch

from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent))

from summarizer import Summarizer, TransformerSummarizer
from summ_pipeline.utils.preprocess_text import sliding_window

def bertSum(text, model):

    chunk_size = 516

    if len(text) > chunk_size:
        chunks = sliding_window(text, chunk_size)
    else:
        chunks = [text]

    for chunk in chunks:
        chunk = clean_text(chunk)
    
    summarized_chunks = []
    for chunk in chunks:
        summarized_chunk = ''.join(model(chunk, min_length=10, max_length=200))
        summarized_chunks.append(summarized_chunk)
    
    return summarized_chunks


#----------- HELPER FUNCTIONS -------------

def clean_text(text):
    text = text.replace('\n',' ')
    text = re.sub(" +", " ", text)
    text = re.sub(r' (?<!\S)\d+(\.\d+)+(?!\S) ', '', text)
    
    dutch_headers = [
    "Inleiding", "Samenvatting", "Achtergrond", "Methodologie", "Resultaten",
    "Discussie", "Conclusie", "Literatuuronderzoek", "Onderzoeksvraag", "Doelstelling",
    "Materiaal en Methoden", "Analyse", "Bespreking van Resultaten", "Implicaties",
    "Toekomstig Onderzoek", "Referenties", "Bijlagen", "Verantwoording", "Abstract",
    "Probleemstelling", "Onderzoeksmethode", "Data-analyse", "Statistische Analyse",
    "Experimenteel Ontwerp", "Case Study", "Literatuuroverzicht", "Conceptueel Kader",
    "Hypothesen", "Onderzoeksopzet", "Onderzoekspopulatie", "Steekproefomvang",
    "Variabelen", "Meetinstrumenten", "Validiteit", "Betrouwbaarheid", "Resultaatinterpretatie",
    "Kritische Reflectie", "Praktische Implicaties", "Beperkingen van het Onderzoek",
    "Aanbevelingen", "Literatuurlijst", "Voetnoten", "Begrippenlijst", "Dankwoord",
    "Voorwoord", "Abstract", "Theoretisch Kader", "Ethische Overwegingen"
    ]
    dutch_head = '|'.join(dutch_headers)

    text = re.sub(dutch_head, '',text)
    text = text.strip()
    return text

"""
if __name__ == "__main__":
    path = "data/data_by_year/2022_rs_data.csv"

    #load data
    rs_df = pd.read_csv(path)

    bert_model = Summarizer()
    idx = 16
    procesverloop = clean_text(rs_df['procesverloop'][idx])
    overwegingen = clean_text(rs_df['overwegingen'][idx])
    beslissing =  clean_text(rs_df['beslissing'][idx])
    proces_summary = ''.join(bert_model(procesverloop, min_length=10, max_length=150))
    overw_summary = ''.join(bert_model(overwegingen, min_length=10, max_length=150))
    beslis_summary = ''.join(bert_model(beslissing, min_length=2, max_length=6))
    print("Length procesverloop " + str(len(proces_summary)) + " Length overwegingen " + str(len(overw_summary))+ " Length beslissing " + str(len(beslis_summary))) 
    print(procesverloop)
    print(proces_summary)
    print("ecli: ", rs_df['ecli' ][idx])
    #print("inhoudsindicatie: ", rs_df['inhoudsindicatie'][idx])
    print("BERT summary: ", proces_summary, '\n' , overw_summary,'\n' , beslis_summary)
    """