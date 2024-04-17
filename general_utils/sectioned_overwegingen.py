import pandas as pd
import re
import multiprocessing
from tqdm import tqdm
import argparse
from pathlib import Path

# Function to concatenate the matched groups (excluding the space)
def replacer(match):
    return f"artikel{match.group(1)}"

# Your text_sectioning function
def text_sectioning(doc, split_patterns, merge_patterns):
    merge_patterns = merge_patterns[0]
    text = re.sub(merge_patterns, replacer, doc)
    super_pattern = "|".join(split_patterns)
    sectioning = re.split(super_pattern, text)
    stripped_list = [s.lstrip() for s in sectioning]
    filtered_list = [s for s in stripped_list if re.match(r'\d+\.', s)]
    return filtered_list #stripped_list #sectioning  # Skip the first part before the first pattern match

# Worker function to be used by multiprocessing.Pool
def worker_func(args):
    text, patterns = args
    return text_sectioning(text, patterns)

# Function to process data in parallel
def parallel_process(func, args_list, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(func, args_list), total=len(args_list)))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data sectioning script')
    parser.add_argument('--input', type=str, default="data/sectioned/2022_rs_data.csv", help="The path to the data CSV file")
    args = parser.parse_args()

    # Load patterns from the file
    split_file = open('general_utils\split_patterns.txt', 'r')
    merge_file = open('general_utils\merge_patterns.txt', 'r')
    split_lines = split_file.readlines()
    split_patterns = [line.strip()[2:-1].replace('\\\\', '\\') for line in split_lines]
    merge_lines = merge_file.readlines()
    merge_patterns = [line.strip()[2:-1].replace('\\\\', '\\') for line in merge_lines]

    # Load the CSV data into a DataFrame
    df_metadata = pd.read_csv(args.input)

    print("Gathering documents...")
    documents = df_metadata['overwegingen'].astype(str).tolist()

    segmented = []
    for i, text in enumerate(documents[:100]):
        if i % 100 == 0:
            print(f"Processing document number: {i} of {len(documents)}")
        sections = text_sectioning(text, split_patterns, merge_patterns)
        sections = [item for item in sections if item is not None]
        segmented.append(sections)


    df_test = df_metadata.copy()
    df_test = df_test.head(100)
    df_test['sections'] = segmented
    print(df_test)
    #df_test = df_test.replace('\n+', ' ', regex=True) # remove \n for excel to view csv correctly
    #print(df_test['sections'].iloc[0])
    df_test.to_csv('sectioned_data_2022_test.csv') # sep = ';' --> let csv use ; as separator
    #df_metadata['sections'] = segmented
    #df_metadata.to_csv('sectioned_data.csv')
