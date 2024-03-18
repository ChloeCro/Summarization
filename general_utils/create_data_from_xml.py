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
    # Your XML processing logic with BeautifulSoup
    # Extract the required tags and build a list
    judgement_list = []

    with open(xml_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')  # Adjust the parser based on your XML format
        # Extract the required tags and append to the result list
        # Example: result_list.append(soup.find('tag_name').text)

        # two lists: 1:titles ; 2:sections
        sections_list = []
        title_list = []
        
        # extraction of the titles in the document
        uitspraak_tag = soup.find("uitspraak")
        if uitspraak_tag:
            titles = uitspraak_tag.find_all("title")
            for title in titles:
                if len(title.text) <= 50 and '[' not in title.text and ']' not in title.text:
                    t = title.text
                    t = t.strip()
                    t = re.sub(r'[0-9!*,)@#%(&$_?.^]', '', t)
                    title_list.append(t)
        
        # extraction of each section in the section list (also save the strings for the 3 separate sections saved)
        sections = soup.find_all("section")
        for sec in sections:
            section_text = ''.join([child.text for child in sec.find_all(is_valid_tag)])
            sections_list.append(section_text)
        try:
            procesverloop = soup.find("section", {"role": "procesverloop"})
            proces_temp = procesverloop.find('title')
            proces_temp.extract()
            procesverloop_text = procesverloop.text.strip()
        except:
            procesverloop_text = ''
        
        try:
            overwegingen = soup.find("section", {"role": "overwegingen"})
            overw_temp = overwegingen.find('title')
            overw_temp.extract()
            overwegingen_text = overwegingen.text.strip()
        except:
            overwegingen_text = ''
            
        try:
            beslissing = soup.find("section", {"role": "beslissing"})
            besl_temp = beslissing.find('title')
            besl_temp.extract()
            beslissing_text = beslissing.text.strip()
        except:
            beslissing_text = ''
        
        try:
            ecli = soup.find("dcterms:identifier").text
        except:
            ecli = ''
        
        try:
            date = soup.find("dcterms:date", {"rdfs:label":"Uitspraakdatum"}).text
        except:
            date = ''

        try:
            inhoud = soup.find("inhoudsindicatie").text
        except:
            inhoud = ''
        
        judgement_list.append(ecli)
        judgement_list.append(date)
        judgement_list.append(inhoud)
        judgement_list.append(title_list)
        judgement_list.append(sections_list)
        judgement_list.append(procesverloop_text)
        judgement_list.append(overwegingen_text)
        judgement_list.append(beslissing_text)

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
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    
    # Set path to XML files
    years = [
    1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932,
    1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952,
    1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972,
    1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
    1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
    2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
    ]

    #year = args.year
    print("start")
    for year in years:
        folder_path = f'unzip_data/{year}'
        xml_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xml')]

        # Process the data
        result_lists = process_files_in_parallel(xml_files)

        # Create df for metadata
        column_names = ['ecli', 'date', 'inhoudsindicatie', 'title_list','sections_list','procesverloop','overwegingen', 'beslissing']
        df = pd.DataFrame(result_lists, columns=column_names)
        print(len(df))

        # Further processing or analysis with the DataFrame
        print(df.head())

        # Optional: Save
        if args.save == True:
            df.to_csv(f'data/data_by_year/{year}_rs_data.csv', index=False)

# python general_utils\create_data_csv.py --year 1905 --save