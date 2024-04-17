import pandas as pd
import re

def get_single_document_list(row):
    """
    (for now) Extracts the texts from the three section columns for each row, 
    combines them and adds them as a full text in a string.

    params:
    row := pandas dataframe row
    """
    procesverloop = str(row['procesverloop'])
    overwegingen = str(row['overwegingen'])
    beslissing = str(row['beslissing'])

    return ' '.join([procesverloop, overwegingen, beslissing])

def get_text(row):
    # For rechtspraak-extractor
    text = str(row['fulltext'])
    return text


def filter_df(df):
    """
    filtering the dataframe based on rules.
    
    Rules:
    1. Remove rows with no or too short reference summaries (inhoudsindicatie text length <= 15)
    """
    df_filtered = df[df['fulltext'].notna() & (df['fulltext'] != '')]
    df_filtered = df_filtered[df_filtered['inhoudsindicatie'].apply(lambda x: len(x.split(' ') if isinstance(x, str) else '') >= 15)]
    return df_filtered

def load_df(path):
    data = pd.read_csv(path)
    return data

def create_csv_from_df():
    pass