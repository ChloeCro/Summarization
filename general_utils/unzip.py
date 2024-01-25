# TODO: add multiprocessing
import os
import zipfile

# Function to unzip files from subdirectories into a folder with the same name
def unzip_files(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file is a zip archive
            if file.endswith(".zip"):
                print("Found zip file:", file_path)

                # Extract the zip file into a folder with the same name as its parent directory
                parent_directory = os.path.basename(root)
                extraction_folder = os.path.join(root, parent_directory)
                os.makedirs(extraction_folder, exist_ok=True)

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_folder)
                    print(f"Extracted to: {extraction_folder}")

# Directory to start searching from
root_directory = "Summarization/data"

# Call the function to unzip files
unzip_files(root_directory)