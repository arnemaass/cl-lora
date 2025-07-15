import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from configilm import util
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
import joblib
from torchvision import transforms
import random

# Paths to the data
datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

# Configuration
COUNTRIES = ['Finland', 'Ireland', 'Serbia', 'Portugal']
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 500
SEED = 42
SAVE_DIR = '~/saved_datasets'

# --- Data preprocessing classes and functions ---
class NormalizeWithStats:
    """SOFTCON normalization with S2A statistics"""
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, img):
        img_np = img.numpy().astype(np.float32)
        # Standard normalization: (x - mean) / std
        img_np = (img_np - self.mean) / self.std
        return torch.from_numpy(img_np).float()

class SelectChannels:
    def __call__(self, img):
        # img is a torch.Tensor [C, H, W]
        return img[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], :, :]

# Training statistics
train_mean = [
    1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
    1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
    1972.62420416, 582.72633433, 1732.16362238, 1247.91870117,
]
train_std = [
    633.15169573, 650.2842772, 712.12507725, 965.23119807,
    948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
    1364.38688993, 472.37967789, 1310.36996126, 1087.6020813,
]

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    SelectChannels(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomChoice([
        transforms.RandomRotation(degrees=(0, 0)),
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.RandomRotation(degrees=(180, 180)),
        transforms.RandomRotation(degrees=(270, 270)),
    ]),
    transforms.RandomResizedCrop(
        size=(128, 128),
        scale=(0.8, 1.0),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    NormalizeWithStats(train_mean, train_std),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    SelectChannels(),
    NormalizeWithStats(train_mean, train_std),
])

def save_country_datasets(country, n_train_samples, n_test_samples, seed, save_dir):
    """Save train and test datasets for a specific country."""
    # Read metadata to get patch IDs for the country
    meta = pd.read_parquet(datapath["metadata_parquet"])
    
    # Get available patches for train split
    train_mask = (meta.country == country) & (meta.split == 'train')
    train_available = meta.loc[train_mask, "patch_id"].tolist()
    
    # Get available patches for test split
    test_mask = (meta.country == country) & (meta.split == 'test')
    test_available = meta.loc[test_mask, "patch_id"].tolist()
    
    # Sample patches
    rng = random.Random(seed)
    train_sampled = set(rng.sample(train_available, k=min(n_train_samples, len(train_available))))
    test_sampled = set(rng.sample(test_available, k=min(n_test_samples, len(test_available))))
    
    # Create dataset instances with appropriate transforms
    train_ds = BENv2DataSet(
        data_dirs=datapath,
        img_size=(14, 128, 128),  # SpectralGPT size
        split='train',
        include_snowy=False,
        include_cloudy=False,
        patch_prefilter=lambda pid: pid in train_sampled,
        transform=train_transform
    )
    
    test_ds = BENv2DataSet(
        data_dirs=datapath,
        img_size=(14, 128, 128),  # SpectralGPT size
        split='test',
        include_snowy=False,
        include_cloudy=False,
        patch_prefilter=lambda pid: pid in test_sampled,
        transform=val_transform
    )
    
    # Create data loaders with fixed batch size
    train_loader = DataLoader(train_ds, batch_size=n_train_samples, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=n_test_samples, shuffle=True)
    
    # Get one batch of samples
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))
    
    # Create save directory for this country
    country_dir = os.path.expanduser(os.path.join(save_dir, country))
    os.makedirs(country_dir, exist_ok=True)
    
    # Save the batches
    torch.save(train_batch, os.path.join(country_dir, 'train_data.pt'))
    torch.save(test_batch, os.path.join(country_dir, 'test_data.pt'))
    
    print(f"Saved datasets for {country}")

def main():
    # Create main save directory
    save_dir = os.path.expanduser(SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save datasets for each country
    for country in COUNTRIES:
        save_country_datasets(
            country=country,
            n_train_samples=TRAIN_SAMPLES,
            n_test_samples=TEST_SAMPLES,
            seed=SEED,
            save_dir=save_dir
        )
    
    print("All datasets saved successfully!")

if __name__ == "__main__":
    main() 