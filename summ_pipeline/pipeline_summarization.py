import os
import re
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np

from multiprocessing import Pool
from functools import partial

from utils.data_loader import get_document_list
from utils.model_utils import load_model
from summ_pipeline.extractive_methods.graph_based_methods import textRank
from summ_pipeline.extractive_methods.clustering_based_methods import k_means
# from summ_pipeline.extractive_methods.heuristic_based_methods import
# from summ_pipeline.extractive_methods.deeplearning_based_methods import

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
        return summarizer.summarize_text(text)

def parallel_process(func, args_list):
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(func, args_list)
    return results

if __name__ == "__main__":
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Summarization script')
    parser.add_argument('--method', type=int, choices=range(1, 6), default=1, help='Specify summarization method (1-5)')
    parser.add_argument('--input', type=string, default="data/input_data.csv", help="The path to the data CSV file")
    args = parser.parse_args()

    # read the data csv as pd dataframe
    df = pd.read_csv(args.input)

    # iterate in parallel over each row of the df to create a list of texts (documents)
    documents = parallel_process(get_document_list,[(row) for row in df.to_dict('records')])  # TODO: expand to text from CSV

    summarizer = Summarization(summarizer_method=args.method)

    # iterate in parallel over each text in the list and summarize them using args.method
    summary_result = parallel_process(summarizer.summarize_wrapper, [(doc, args.method) for doc in documents])

    print(summary_result)
    # TODO: Expand by adding all the summaries to a CSV

# 1: TextRank
# 2: k-means clustering
# 3: extractive BERT (non finetuned model)
# 4: extractive GPT2 (non finetuned model)
    
# TO CREATE requirements.txt RUN FOLLOWING:
    # pip freeze > requirements.txt
