import pandas as pd

def get_document_list(row):
    """
    (for now) Extracts the texts from the three section columns for each row, 
    combines them and adds them as a full text in a string.

    params:
    row := pandas dataframe row
    """
    procesverloop = row['procesverloop']
    overwegingen = row['overwegingen']
    beslissing = row['beslissing']

    return ' '.join([procesverloop, overwegingen, beslissing])
