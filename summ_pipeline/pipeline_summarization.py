import os
import re
import sys
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np

from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent))

from general_utils.data_loader import get_single_document_list, filter_df, create_csv_from_df
from general_utils.model_utils import load_model
from summ_pipeline.extractive_methods.graph_based_methods import textRank
from summ_pipeline.extractive_methods.clustering_based_methods import k_means
# from summ_pipeline.extractive_methods.heuristic_based_methods import
# from summ_pipeline.extractive_methods.deeplearning_based_methods import bertSum, gptSum

class Summarization():

    def __init__(self, method):
        self.method = method 

        #if self.requires_model(method): # will prob not need this
            # is_model specifies if a model is required for a summarization method, so we can load it here --> will prob not need this
        #    self.summarizer_model = load_model()


    def requires_model(self, method_number):
        # Implement logic to determine if the method requires a model
        # For example, you might have a list of methods that require a model
        methods_requiring_model = [2, 4]  # TODO: Set this to correct values
        return method_number in methods_requiring_model

    def summarize_text(self, text):
        # result should be list of sentences!

        result = []
        # run the summarization using the method
        if self.method == 1:
            result = textRank(text) # result is a list of top sentences!!
        elif self.method == 2:
            return k_means(text) # add param for number of clusters?
        #elif self.summarizer_method == 3:
        #    return method_3_summary(text)
        # Add more methods as needed
        else:
            raise ValueError(f"Unsupported summarization method: {self.method}")

        return result
    
    @staticmethod
    def summarize_wrapper(text, method): 
        summarizer = Summarization(method)
        result = summarizer.summarize_text(text)
        return result

def parallel_process(func, args_list): 
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    with multiprocessing.Pool(processes=num_processes) as pool:
        #results = list(tqdm(pool.imap(func, args_iterable), total=len(args_iterable)))
        results = pool.starmap(func, args_list)
    return results

if __name__ == "__main__":
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Summarization script')
    parser.add_argument('--method', type=int, choices=range(1, 6), default=1, help='Specify summarization method (1-5)')
    parser.add_argument('--input', type=str, default="data/input_data.csv", help="The path to the data CSV file")
    args = parser.parse_args()

    # read the data csv as pd dataframe and filter df by reference summaries
    df = pd.read_csv(args.input)
    df = filter_df(df)

    # iterate in parallel over each row of the df to create a list of texts (documents)
    print("Gathering documents...")
    documents = parallel_process(get_single_document_list,[(row,) for row in df.to_dict('records')])  # TODO: expand to text from CSV

    #summarizer = Summarization(method=args.method)
    
    #raise RuntimeError('Intentionally stopping the code here for debugging purposes.')

    # iterate in parallel over each text in the list and summarize them using args.method
    print(f"Summarizing using method {args.method}...")
    summary_result = parallel_process(Summarization.summarize_wrapper, [(doc, args.method) for doc in documents])
    #summary_result = summarizer.summarize_text(documents[0])

    print(summary_result)
    df['gen_summary'] = summary_result
    print(df)
    # TODO: Expand by adding all the summaries to a CSV

# EXTRACTIVE
# 1: TextRank
# 2: K-means clustering
# 3: BERT (non finetuned model)
# 4: GPT2 (non finetuned model)
# -----------------------------------------
# ABSTRACTIVE
# 5: T5
    
# TO CREATE requirements.txt RUN FOLLOWING:
    # pip freeze > requirements.txt

# python summ_pipeline/pipeline_summarization.py --method 1 --input data/2022_rs_data.csv
