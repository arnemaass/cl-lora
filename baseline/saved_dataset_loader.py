import os
import torch
from torch.utils.data import Dataset, DataLoader

class SavedCountryDataset(Dataset):
    """Dataset class for loading saved country data."""
    def __init__(self, data_tuple):
        self.images = data_tuple[0]  # Assuming first element is images
        self.labels = data_tuple[1]  # Assuming second element is labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_country_datasets(countries, save_dir='saved_datasets'):
    """
    Load saved datasets for specified countries.
    
    Args:
        countries (list): List of country names to load
        save_dir (str): Directory where datasets are saved
        
    Returns:
        tuple: Lists of train and test datasets
    """
    train_datasets = []
    test_datasets = []
    
    for country in countries:
        country_dir = os.path.join(save_dir, country)
        
        # Load train data
        train_data = torch.load(os.path.join(country_dir, 'train_data.pt'))
        train_ds = SavedCountryDataset(train_data)
        train_datasets.append(train_ds)
        
        # Load test data
        test_data = torch.load(os.path.join(country_dir, 'test_data.pt'))
        test_ds = SavedCountryDataset(test_data)
        test_datasets.append(test_ds)
        
        print(f"Loaded datasets for {country}")
    
    return train_datasets, test_datasets 