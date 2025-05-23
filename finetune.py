import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from configilm import util
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule

# Set random seeds for reproducibility
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
generator = torch.Generator().manual_seed(seed)

datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

# --- Normalization function as provided ---
class SelectFirst2Channels:
    def __call__(self, img):
        # img is a torch.Tensor [C, H, W]
        return img[:2, :, :]

def normalize(img, mean, std):
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# Temporary transform to get raw data for stats
temp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    SelectFirst2Channels(),
])

# Load dataset with temp transform for stats
full_dataset = BENv2_DataSet.BENv2DataSet(
    data_dirs=datapath,
    max_len=1000,
    transform=temp_transform
)

# Split before computing stats
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

# Compute mean and std over training set (first 2 channels)
def compute_mean_std(dataset, num_samples=200):
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    n = 0
    mean = 0
    std = 0
    for i, (img, _) in enumerate(loader):
        img = img.float()
        mean += img.mean(dim=[0, 2, 3])
        std += img.std(dim=[0, 2, 3])
        n += 1
        if n * loader.batch_size >= num_samples:
            break
    mean /= n
    std /= n
    return mean.numpy(), std.numpy()

train_mean, train_std = compute_mean_std(train_dataset)
print("Train mean:", train_mean)
print("Train std:", train_std)

# --- Final transform with normalization using train stats ---
class NormalizeWithStats:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, img):
        img_np = img.numpy()
        img_np = normalize(img_np, self.mean, self.std)
        img_np = img_np.astype(np.float32) / 255.0  # convert to float32 and scale
        return torch.from_numpy(img_np)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    SelectFirst2Channels(),
    NormalizeWithStats(train_mean, train_std),
])

# --- Country-specific train/val split function ---
meta = pd.read_parquet(datapath["metadata_parquet"])

def load_country_train_val(
    country: str,
    n_samples: int,
    seed: int = None,
    train_frac: float = 0.8,
    img_size=(14, 120, 120),
    include_snowy=False,
    include_cloudy=False,
):
    """
    Samples n_samples patches from the TRAIN split of `country`,
    then does an internal 80/20 train/val split (both from the original TRAIN data).
    Returns (train_ds, val_ds).
    """
    mask = (meta.country == country) & (meta.split == "train")
    available = meta.loc[mask, "patch_id"].tolist()
    if not available:
        raise ValueError(f"No TRAIN patches found for country={country!r}")
    if n_samples > len(available):
        raise ValueError(
            f"Requested {n_samples} samples but only {len(available)} TRAIN patches available for {country!r}"
        )
    rng = random.Random(seed)
    sampled = rng.sample(available, k=n_samples)
    split_at = int(n_samples * train_frac)
    train_ids = set(sampled[:split_at])
    val_ids   = set(sampled[split_at:])
    def _make_ds(keep_ids):
        return BENv2_DataSet.BENv2DataSet(
            data_dirs=datapath,
            img_size=img_size,
            split="train",
            include_snowy=include_snowy,
            include_cloudy=include_cloudy,
            patch_prefilter=lambda pid: pid in keep_ids,
            transform=transform,
        )
    train_ds = _make_ds(train_ids)
    val_ds   = _make_ds(val_ids)
    return train_ds, val_ds

# --- Usage: get train/val splits for a country ---
ds_train, ds_val = load_country_train_val("Ireland", n_samples=4874, seed=seed)

# DataLoaders for train and validation
train_loader = DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=4)

# --- Model setup ---
sys.path.append('/home/arne/softcon')
from models.dinov2 import vision_transformer as dinov2_vits

model_vits14 = dinov2_vits.__dict__['vit_small'](
    img_size=224,
    patch_size=14,
    in_chans=2,
    block_chunks=0,
    init_values=1e-5,
    num_register_tokens=0,
)

# load pretrained weights
ckpt_vits14 = torch.load('/faststorage/softcon/pretrained/B2_vits14_softcon.pth', map_location='cpu')
model_vits14.load_state_dict(ckpt_vits14)

# Wrap with LoRA
sys.path.append('/home/arne/LoRA-ViT')
from lora import LoRA_ViT_timm

lora_model = LoRA_ViT_timm(model_vits14, r=4, alpha=16)

# Add classification head
num_classes = 19
classifier = nn.Linear(model_vits14.embed_dim, num_classes)
lora_with_head = nn.Sequential(lora_model, classifier)

print(lora_with_head)
print('\nNumber of trainable parameters:', sum(p.numel() for p in lora_with_head.parameters() if p.requires_grad))

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_with_head.to(device)

# --- Training loop ---
epochs = 10
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_with_head.parameters()), lr=1e-4)

for epoch in range(epochs):
    lora_with_head.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = lora_with_head(imgs)
        if labels.dtype != torch.float:
            labels = labels.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # Validation loss
    lora_with_head.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if labels.dtype != torch.float:
                labels = labels.float()
            outputs = lora_with_head(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

save_path = "/faststorage/softcon/finetuning/finland.pth"
torch.save(lora_with_head.state_dict(), save_path)
print(f"Model saved to {save_path}")