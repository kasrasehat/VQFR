import os
import shutil
import pandas as pd
import glob

# Define paths
csv_file_path = '/home/user1/kasra/pycharm-projects/VQFR/high_quality_documents.csv'
destination_folder = '/mnt/drive/cleaned_phase_2_celeb_ds/high_quality_completed1/'

# Create destination folder if it does not exist
os.makedirs(destination_folder, exist_ok=True)

# Get list of existing files in the destination folder
existing_files = glob.glob(os.path.join(destination_folder, "*"))
if existing_files:
    # Extract numbers from existing file names
    existing_numbers = [int(os.path.splitext(os.path.basename(f))[0]) for f in existing_files if os.path.basename(f).split('.')[0].isdigit()]
    # Start counter after the highest existing number
    counter = max(existing_numbers) + 1
else:
    counter = 1

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Iterate through each row and copy files with quality_value > 93
for index, row in df.iterrows():
    saved_path = '/mnt/drive' + row['saved_path']
    quality_value = row['quality_value']
    
    # If quality_value is above 93, copy the file to the destination folder
    if quality_value > 84:
        if os.path.exists(saved_path):
            # Get the original file extension
            _, file_extension = os.path.splitext(saved_path)
            # Define the new file name with zero-padded counter
            new_file_name = f"{counter:05d}{file_extension}"
            destination_path = os.path.join(destination_folder, new_file_name)
            
            # Copy and rename the file
            shutil.copy(saved_path, destination_path)
            
            # Increment the counter
            counter += 1

            
            
            
            
            