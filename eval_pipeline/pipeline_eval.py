import os
import re
import sys
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent))

from general_utils.data_helper import get_single_document_list, filter_df, create_csv_from_df
from eval_pipeline.evaluation_methods import rouge_method, bertscore, llm_method

class Evaluation():

    def __init__(self, method):
        self.method = method 

    def requires_model(self, method_number):
        # Implement logic to determine if the method requires a model
        # For example, you might have a list of methods that require a model
        methods_requiring_model = [3]  # TODO: Set this to correct values
        return method_number in methods_requiring_model

    def evaluate(self, reference, summary):
        # result should be list of sentences!

        result = []
        # run the evaluation using the method
        if self.method == 1:
            result = rouge_method(reference, summary) # result is a list of top sentences!!
        elif self.method == 2:
            result = bertscore(reference, summary) # add param for number of clusters?
        elif self.method == 3:
            result = llm_method(reference, summary)
        # Add more methods as needed
        else:
            raise ValueError(f"Unsupported evaluation method: {self.method}")

        return result #return list of evaluation results
    
    @staticmethod
    def evaluator_wrapper(reference, summary, method): 
        evaluator = Evaluation(method)
        result = evaluator.evaluate(reference, summary) 
        return result

def parallel_process(func, args_list): 
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(func, args_list)
    return results

if __name__ == "__main__":
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--method', type=int, choices=range(1, 6), default=1, help='Specify evaluation method (1-5)')
    parser.add_argument('--input', type=str, default="data/data_by_year/2022_rs_data.csv", help="The path to the data CSV file")
    #parser.add_argument('--multi', action='store_true', help="Use multiprocessing?")
    args = parser.parse_args()

    # read the data csv as pd dataframe and filter df by reference summaries
    df = pd.read_csv(args.input)
    
    #try it with a number of cases
    df = df.head(10)

    # iterate in parallel over each row of the df to create a list of texts (documents)
    print("Gathering references and summaries...")
    references = df['inhoudsindicatie'].tolist()  # TODO: expand to text from CSV
    summaries = df['gen_summary'].tolist()

    evaluator = Evaluation(args.method)
    #TODO: expand to get evaluation results
    print("Evaluating...")
    
    results = []
    for i in range(0, 10):
        result = evaluator.evaluate(references[i], summaries[i])
        results.append(result)
    
    #results = parallel_process(Evaluation.evaluator_wrapper, [(ref, summ, args.method) for ref, summ in zip(references, summaries)])
    df = pd.DataFrame(results)
    print(df)

# EVALUATIONS
# 1: ROUGE

# CSV FORMAT:
    # HAVE 2 COLUMNS OF "REFERENCE" & "PREDICTION" !!!!!!

# python eval_pipeline/pipeline_eval.py --method 1 --input bert_test_results_1.csv