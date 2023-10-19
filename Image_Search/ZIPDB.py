import os
import zipfile

# Specify your directory here
dir_path = "C:\\Users\\DGU_ICE\\AI_Trademark_IMG"

# Loop over every file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.zip'):
        # Construct full file path
        filepath = os.path.join(dir_path, filename)
        
        # Open the zip file
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            # Extract all files to a directory named after the zip file (without the .zip extension)
            extract_dir = os.path.join(dir_path, filename[:-4])
            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)

        # Delete the original zip file
        os.remove(filepath)
