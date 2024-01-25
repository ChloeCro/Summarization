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
    parser.add_argument('--year', type=str, help='the year we want to process')
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    
    # Set path to XML files
    year = args.year
    print("start")
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
        df.to_csv(f'{year}_rs_data.csv', index=False)