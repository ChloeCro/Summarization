import os
import re
import sys
import ast
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import rechtspraak_extractor as rex
from pathlib import Path
from tqdm import tqdm


script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent))

from general_utils.data_helper import get_single_document_list, get_text

def text_sectioning(doc, patterns):
    # should return a list containing the sections

    #print(patterns)
    super_pattern = "|".join(patterns)
    #print(super_pattern)

    sectioning = re.split(super_pattern, doc)

    return sectioning[1:]

def worker_func(args):
    # Assuming 'func' is defined at the top level or is importable
    func, args = args[0], args[1:]
    return func(*args)

def parallel_process(func, args_list):
    num_processes = multiprocessing.cpu_count() 

    # Prepare a new args_list that includes the target function as the first element of each tuple
    new_args_list = [(func,) + args for args in args_list]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(worker_func, new_args_list), total=len(args_list)):
            results.append(result)
    return results

"""
def parallel_process(func, args_list): 
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(func, args_list)
    return results
"""

if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Data sectioning script')
    parser.add_argument('--input', type=str, default="data/data_by_year/2022_rs_data.csv", help="The path to the data CSV file")
    #parser.add_argument('--multi', action='store_true', help="Use multiprocessing?")
    args = parser.parse_args()

    # acquire the data using rechtspraak extractor
    #df = rex.get_rechtspraak(max_ecli=1000, sd='2022-03-01', ed='2022-05-01', save_file='n')
    #df_metadata = rex.get_rechtspraak_metadata(save_file='n', dataframe=df)

    df_metadata = pd.read_csv(args.input)
    pattern_file = open('general_utils\patterns.txt', 'r')
    lines = pattern_file.readlines()
    patterns = [line.strip()[2:-1].replace('\\\\', '\\') for line in lines]

    print("Gathering documents...")
    documents = parallel_process(get_text,[(row,) for row in df_metadata.to_dict('records')])

    segmented = []
    for i, text in enumerate(documents[:100]):
        if i % 100 == 0:
            print(f"Processing document number: {i} of {len(documents)}")
        sections = text_sectioning(text, patterns)
        sections = [item for item in sections if item is not None]
        segmented.append(sections)

    #print(segmented)
    df_test = df_metadata.copy()
    df_test = df_test.head(100)
    df_test['sections'] = segmented
    print(df_test)
    df_test = df_test.replace('\n+', ' ', regex=True) # remove \n for excel to view csv correctly
    #print(df_test['sections'].iloc[0])
    df_test.to_csv('sectioned_data.csv') # sep = ';' --> let csv use ; as separator
    #df_metadata['sections'] = segmented
    #df_metadata.to_csv('sectioned_data.csv')
    # do some magical dataframe filtering to only get entries with full-text and other stuff
    

    # section the data

# python general_utils\create_sectioned_data.py --input data/data_by_year/2022_rs_data.csv