import zipfile
import os

def unzip_file(zip_path, extract_to):
    """
    Unzips a file to the specified directory.

    Args:
        zip_path (str): Path to the .zip file.
        extract_to (str): Directory where files will be extracted.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"The file {zip_path} is not a valid zip file.")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted {len(zip_ref.namelist())} files to {extract_to}")

# Example usage
zip_path = '/data/elo/khoatn/PathAttack-Recontruction/POPOP_location/annotation/lfw_dataset.zip'
extract_to = './lfw_dataset'

os.makedirs(extract_to, exist_ok=True)
unzip_file(zip_path, extract_to)

