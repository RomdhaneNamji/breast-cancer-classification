"""Download the dataset into data/raw/.   Usage: python -m src.download_data"""
from src.data import download_dataset

if __name__ == "__main__":
    print(f"Saved dataset to {download_dataset()}")
