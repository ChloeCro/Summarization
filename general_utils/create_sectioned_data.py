import os
import re
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import rechtspraak_extractor as rex

def text_sectioning(doc):
    pass

if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Data sectioning script')
    parser.add_argument('--method', type=int, choices=range(1, 6), default=1, help='Specify summarization method (1-5)')
    parser.add_argument('--input', type=str, default="data/data_by_year/2022_rs_data.csv", help="The path to the data CSV file")
    parser.add_argument('--multi', action='store_true', help="Use multiprocessing?")
    args = parser.parse_args()

    # acquire the data using rechtspraak extractor
    df = rex.get_rechtspraak(max_ecli=1000, sd='2022-03-01', ed='2022-05-01', save_file='n')
    df_metadata = rex.get_rechtspraak_metadata(save_file='n', dataframe=df)

    # do some magical dataframe filtering to only get entries with full-text and other stuff


    # section the data

# python general_utils\create_data_csv.py --year 1905 --save