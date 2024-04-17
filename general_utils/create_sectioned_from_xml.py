import os
import re
import string
import argparse
import multiprocessing
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def is_valid_tag(tag):
    return tag.name != 'title'

def process_xml(xml_file):
    with open(xml_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')
        
        # Initialize variables
        procesverloop_text, overwegingen_text, beslissing_text = '', '', ''
        ecli, date, inhoud = '', '', ''
        
        # Extract global information
        ecli_tag = soup.find("dcterms:identifier")
        date_tag = soup.find("dcterms:date", {"rdfs:label": "Uitspraakdatum"})
        inhoud_tag = soup.find("inhoudsindicatie")
        if ecli_tag: ecli = ecli_tag.text
        if date_tag: date = date_tag.text
        if inhoud_tag: inhoud = inhoud_tag.text

        # Process each section
        sections = soup.find_all("section")
        for sec in sections:
            role = sec.get('role')
            if role == 'overwegingen' or role is None:
                # Convert the entire section into string and process tags for 'overwegingen'
                section_text = ''.join(str(child) for child in sec.contents)
            else:
                # Standard process for other roles
                section_text = ' '.join(text for text in sec.stripped_strings)
            
            # Append text based on the role of the section
            if role == 'procesverloop':
                procesverloop_text += (' ' + section_text if procesverloop_text else section_text)
            elif role == 'beslissing':
                beslissing_text += (' ' + section_text if beslissing_text else section_text)
            else:  # This will now be only 'overwegingen' or sections without a role
                overwegingen_text += (' ' + section_text if overwegingen_text else section_text)

        # Check if 'procesverloop' and 'beslissing' are present
        if not procesverloop_text or not beslissing_text:
            return None  # Skip file if critical sections are missing

        # Compile all extracted information into a list
        judgement_list = [ecli, date, inhoud, procesverloop_text, overwegingen_text, beslissing_text]

        return judgement_list

    
def process_files_in_parallel(files):
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    pool = multiprocessing.Pool(processes=num_processes)

    # Use the multiprocessing pool to process the XML files in parallel
    print("start multiprocessing here")
    result_lists = pool.map(process_xml, files)
    print(len(result_lists))
    # Close the pool and wait for the worker processes to finish
    pool.close()
    pool.join()

    return result_lists

if __name__ == '__main__':
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    #parser.add_argument('--year', type=str, help='the year we want to process or \'all\' if we want to process all years.')
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()
    
    # Set path to XML files
    """years = [
    1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932,
    1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952,
    1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972,
    1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
    1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
    2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
    ]"""
    years = [2022]

    #year = args.year
    print("start")
    for year in years:
        folder_path = f'unzip_data/{year}'
        xml_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xml')]

        # Process the data
        result_lists = process_files_in_parallel(xml_files)
        filtered_results = [result for result in result_lists if result is not None]

        # Create df for metadata
        column_names = ['ecli', 'date', 'inhoudsindicatie', 'procesverloop','overwegingen', 'beslissing']
        df = pd.DataFrame(filtered_results, columns=column_names)
        print(len(df))

        # Further processing or analysis with the DataFrame
        print(df.head())

        # Optional: Save
        if args.save == True:
            df.to_csv(f'data/sectioned/{year}_rs_data_test.csv', index=False)

# python general_utils\create_data_csv.py --year 1905 --save
# python general_utils/create_sectioned_from_xml.py --save
