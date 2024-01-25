import os
import re
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np

from multiprocessing import Pool
from functools import partial

from utils.model_utils import load_model
from summ_pipeline.extractive_methods.graph_based_methods import textRank
# from summ_pipeline.methods.clustering_based_methods import
# from summ_pipeline.methods.heuristic_based_methods import

class Summarization():

    def __init__(self, method):
        self.method = method 
        if self.requires_model(method): # will prob not need this
            # is_model specifies if a model is required for a summarization method, so we can load it here --> will prob not need this
            self.summarizer_model = load_model()

    def requires_model(self, method_number):
        # Implement logic to determine if the method requires a model
        # For example, you might have a list of methods that require a model
        methods_requiring_model = [2, 4]  # TODO: Set this to correct values
        return method_number in methods_requiring_model

    def summarize_text(self, text):

        result = []
        # run the summarization using the method
        if self.method == 1:
            result = textRank(text) # result is a list of top sentences!!
        #elif self.summarizer_method == 2:
        #    return method_2_summary(text)
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
    
    def parallel_summarization(self, documents):
        func_with_method = partial(Summarization.summarize_wrapper, method=self.method)

        with Pool(processes=4) as pool:  # Adjust the number of processes as needed
            results = pool.starmap(func_with_method, [(doc,) for doc in documents])
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarization script')

    parser.add_argument('--method', type=int, choices=range(1, 6), default=1, help='Specify summarization method (1-5)')

    args = parser.parse_args()

    summarizer = Summarization(summarizer_method=args.method)

    # Example usage:
    documents = "Your input text goes here."  # TODO: expand to text from CSV
    summary_result = summarizer.parallel_summarization(documents)
    print(summary_result)
    # TODO: Expand by adding all the summaries to a CSV

# 1: TextRank
# 2: k-means clustering
    
# TO CREATE requirements.txt RUN FOLLOWING:
    # pip freeze > requirements.txt
