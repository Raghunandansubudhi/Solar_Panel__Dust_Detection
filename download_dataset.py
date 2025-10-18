import kaggle
import time
import sys
from zipfile import ZipFile
import os

def download_with_retry(dataset_name, max_retries=3):
    """Download dataset with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Downloading {dataset_name}...")
            kaggle.api.dataset_download_files(dataset_name, path='.', unzip=False)
            print("Download completed successfully!")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("All retry attempts failed.")
                return False

def extract_zip(zip_path):
    """Extract zip file"""
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Successfully extracted {zip_path}")
        return True
    except Exception as e:
        print(f"Failed to extract {zip_path}: {str(e)}")
        return False

if __name__ == "__main__":
    dataset_name = "hemanthsai7/solar-panel-dust-detection"
    
    # Download dataset
    if download_with_retry(dataset_name):
        # Extract the downloaded zip file
        zip_filename = "solar-panel-dust-detection.zip"
        if os.path.exists(zip_filename):
            extract_zip(zip_filename)
            # Clean up zip file after extraction
            os.remove(zip_filename)
            print("Dataset download and extraction completed!")
        else:
            print("Zip file not found after download")
    else:
        print("Failed to download dataset after multiple attempts")