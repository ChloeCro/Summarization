import os
import multiprocessing
from bs4 import BeautifulSoup
import argparse

def process_xml(xml_file):
    # Your XML processing logic with BeautifulSoup
    found = False

    with open(xml_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')
        sections = soup.find_all("emphasis")
        overweging = soup.find_all("section", {"role": "overwegingen"})
        
        if len(sections) > 0 and len(overweging) > 0:
            found = True

    return found

def worker(xml_file):
    # This is the worker function that processes a single XML file.
    return process_xml(xml_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()

    years = [2022]
    print("start")

    for year in years:
        folder_path = f'unzip_data/{year}'
        xml_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xml')]

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # Use map to apply the worker function to all items in xml_files
            results = pool.map(worker, xml_files)

        # Results is a list of True/False values indicating whether each file contains the required sections
        count = sum(results)  # Count how many files contained the required sections

        print(f'{count}/{len(xml_files)}')

