import os
import re
import sys
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import rechtspraak_extractor as rex
from pathlib import Path


script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent))

from general_utils.data_helper import get_single_document_list, get_text

def text_sectioning(doc, patterns):
    # should return a list containing the sections

    print(patterns)
    super_pattern = "|".join(patterns)
    print(super_pattern)

    sectioning = re.split(super_pattern, doc)

    return sectioning

def parallel_process(func, args_list): 
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(func, args_list)
    return results

if __name__ == '__main__':
    # create the argument parser, add the arguments
    #parser = argparse.ArgumentParser(description='Data sectioning script')
    #parser.add_argument('--method', type=int, choices=range(1, 6), default=1, help='Specify summarization method (1-5)')
    #parser.add_argument('--input', type=str, default="data/data_by_year/2022_rs_data.csv", help="The path to the data CSV file")
    #parser.add_argument('--multi', action='store_true', help="Use multiprocessing?")
    #args = parser.parse_args()

    # acquire the data using rechtspraak extractor
    #df = rex.get_rechtspraak(max_ecli=1000, sd='2022-03-01', ed='2022-05-01', save_file='n')
    #df_metadata = rex.get_rechtspraak_metadata(save_file='n', dataframe=df)

    df_metadata = pd.read_csv('rechtspraak_data.csv')
    pattern_file = open('general_utils\patterns.txt', 'r')
    lines = pattern_file.readlines()
    patterns = [line.strip()[2:-1].replace('\\\\', '\\') for line in lines]

    print("Gathering documents...")
    documents = parallel_process(get_text,[(row,) for row in df_metadata.to_dict('records')])

    segmented = []
    for text in documents[:1]:
        print(text)
        sections = text_sectioning(text, patterns)
        segmented.append(sections)

    print(segmented)
    #print(df_metadata)



    #df_metadata.to_csv('rechtspraak_data.csv')
    # do some magical dataframe filtering to only get entries with full-text and other stuff
    

    # section the data

# python general_utils\create_data_csv.py --year 1905 --save