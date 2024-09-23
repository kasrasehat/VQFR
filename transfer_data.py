import os
import shutil
import pandas as pd

# Define paths
csv_file_path = '/home/user1/kasra/pycharm-projects/VQFR/high_quality_documents.csv'
destination_folder = '/mnt/drive/cleaned_phase_2_celeb_ds/high_quality/'

# Create destination folder if it does not exist
os.makedirs(destination_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Iterate through each row and copy files with quality_value > 90
for index, row in df.iterrows():
    saved_path = '/mnt/drive' + row['saved_path']
    quality_value = row['quality_value']
    
    # If quality_value is above 90, copy the file to the destination folder
    if quality_value > 93:
        if os.path.exists(saved_path):
            shutil.copy(saved_path, destination_folder)