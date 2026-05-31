import os
print("Initializing the classic Kaggle API...")

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
except Exception as e:
    print(f"\n Authentication failed: {e}")
    exit()

try:
    # Downloads the entire dataset and unzips it directly (unzip=True)
    api.dataset_download_files('nih-chest-xrays/data', path='.', unzip=True)
    print("\nDataset downloaded and unzipped")
except Exception as e:
    print(f"\n Error: {e}")