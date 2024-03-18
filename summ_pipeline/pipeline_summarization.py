import os
import re
import sys
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import datetime
import psutil
import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from pathlib import Path
from summarizer import Summarizer, TransformerSummarizer

script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent))

from general_utils.data_helper import get_single_document_list, filter_df, create_csv_from_df, get_text
from general_utils.model_utils import load_model
from summ_pipeline.utils.preprocess_text import remove_info
from summ_pipeline.extractive_methods.graph_based_methods import textRank
from summ_pipeline.extractive_methods.clustering_based_methods import k_means
# from summ_pipeline.extractive_methods.heuristic_based_methods import
from summ_pipeline.extractive_methods.deeplearning_based_methods import bertSum #, gptSum
from old.bert_method import bert
#from summ_pipeline.abstractive_methods.methods import bart, t5

class Summarization():

    def __init__(self, method):
        self.method = method 

        #if self.requires_model(method): # will prob not need this
            # is_model specifies if a model is required for a summarization method, so we can load it here --> will prob not need this
        #    self.summarizer_model = load_model()


    def requires_model(self, method_number):
        # Implement logic to determine if the method requires a model
        # For example, you might have a list of methods that require a model
        methods_requiring_model = [3]  # TODO: Set this to correct values
        return method_number in methods_requiring_model

    def summarize_text(self, text, param, model=None):
        # result should be list of sentences!

        # remove uitspraak info text from top of document
        text = remove_info(text)

        result = []
        # run the summarization using the method
        if self.method == 1:
            result = textRank(text, param[0]) # result is a list of top sentences!!
        elif self.method == 2:
            result = k_means(text) # add param for number of clusters?
        elif self.method == 3:
            #result = bert(text)
            model = Summarizer()
            result = bertSum(text, model)
        elif self.method == 4:
            result = bart(text)
        # Add more methods as needed
        else:
            raise ValueError(f"Unsupported summarization method: {self.method}")

        return ' '.join(result) #result

    @staticmethod
    def summarize_wrapper(text, method, param): 
        model = Summarizer()
        summarizer = Summarization(method)
        result = summarizer.summarize_text(text, param, model)
        return result
"""
Added worker_func and new parallel_process to add progress bar
"""
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
# old parallel_process before progress bar
def parallel_process(func, args_list): 
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.starmap(func, args_list), total=len(args_list)))
        #results = pool.starmap(func, args_list)
    return results
"""
if __name__ == "__main__":
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Summarization script')
    parser.add_argument('--method', type=int, choices=range(1, 6), default=1, help='Specify summarization method (1-5)')
    parser.add_argument('--input', type=str, default="data/data_by_year/2022_rs_data.csv", help="The path to the data CSV file")
    parser.add_argument('--multi', action='store_true', help="Use multiprocessing?")
    parser.add_argument('--n_sent', type=int, default=0, help="Required for top sentences methods: how many sentences returned / how many clusters used")
    parser.add_argument('--n_cases', type=int, required=False, help="Number of cases to process")

    args = parser.parse_args()

    param = [args.n_sent]

    # read the data csv as pd dataframe and filter df by reference summaries
    df = pd.read_csv(args.input)
    df = filter_df(df)
    
    #try it with a number of cases
    #df = df.head(100)
    print(psutil.cpu_count(logical=True))

    # iterate in parallel over each row of the df to create a list of texts (documents)
    print("Gathering documents...")
    documents = parallel_process(get_text,[(row,) for row in df.to_dict('records')])  # TODO: expand to text from CSV
    
    print(f"Number of documents: {len(documents)}")
    #raise RuntimeError('Intentionally stopping the code here for debugging purposes.')
    summarizer = Summarization(args.method)

    # iterate in parallel over each text in the list and summarize them using args.method
    print(f"Summarizing using method {args.method}...")
    if summarizer.requires_model(args.method):
        print("Model required...")
        #summary_result = summarizer.summarize_text(documents, param)
        bert_model = Summarizer()
        summary_result = []
        for id, doc in enumerate(documents[:args.n_cases]):
            print("Number " + str(id) + " of " + str(len(documents[:args.n_cases])))
            summary = summarizer.summarize_text(doc, param, bert_model)
            summary_result.append(summary)
    elif args.multi == False:
        summary_result = []
        summary = summarizer.summarize_text(documents[0], param)
        summary_result.append(summary)
    else:
        print("Multiprocessing...")
        summary_result = parallel_process(Summarization.summarize_wrapper, [(doc, args.method, param) for doc in documents])
        #summary_result = summarizer.summarize_text(documents[0])

    print(summary_result)
    df_copy = df
    df_copy = df_copy.head(args.n_cases)
    df_copy['prediction'] = summary_result
    #print(df_copy)

    # Get the current date
    current_date = datetime.datetime.now()

    # Format the date as 'dd_mm_yyyy'
    date_str = current_date.strftime('%d_%m_%Y')

    # Get method name for folder:
    if args.method == 1:
        path = "results/results_textrank/"
    elif args.method == 2:
        path = "results/results_kmeans/"
    elif args.method == 3:
        path = "results/results_bertextract/"
    elif args.method == 4:
        path = "results/results_bart/"
    else:
        raise ValueError(f"Unsupported summarization method: {args.method}")

    # Create the filename with the current date
    filename = path + f'result_{date_str}.csv'

    df_copy.to_csv(filename, index=False)



# EXTRACTIVE
# 1: TextRank
# 2: K-means clustering
# 3: BERT (non finetuned model)
# 4: GPT2 (non finetuned model)
# -----------------------------------------
# ABSTRACTIVE
# 5: BART
# 6: T5
    
# TO CREATE requirements.txt RUN FOLLOWING:
    # pip freeze > requirements.txt

# python summ_pipeline/pipeline_summarization.py --method 1 --input data/2022_rs_data.csv
# python summ_pipeline/pipeline_summarization.py --method 1 --input data/fulltext_by_year/2022_rs_data.csv --n_sent 25 --multi
# python summ_pipeline/pipeline_summarization.py --method 3 --input data/fulltext_by_year/2022_rs_data.csv 