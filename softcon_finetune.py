import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import csv
from tqdm import tqdm

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

# SOFTCON PRETRAINING band information
ALL_BANDS_S2_L2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

# Band statistics: mean & std (calculated from 50k data)
S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 
           2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]

S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 
          1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages



# Remove the old normalize function and replace with standard normalization
class NormalizeWithStats:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)  # Shape: (C, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)    # Shape: (C, 1, 1)
    
    def __call__(self, img):
        img_np = img.numpy().astype(np.float32)  # Ensure float32
        # Standard normalization: (x - mean) / std
        # This properly uses your S2A statistics for zero-mean, unit-variance normalization
        img_np = (img_np - self.mean) / self.std
        return torch.from_numpy(img_np).float()  # Explicitly convert to float32

# Use 12-channel S2A statistics directly
train_mean, train_std = np.array(S2A_MEAN), np.array(S2A_STD)
print("Using pre-calculated S2A stats for 12 Sentinel-2 bands:")
print("Train mean:", train_mean)
print("Train std:", train_std)

# Drop the first 2 SAR channels to keep only Sentinel-2 bands
class DropSARChannels:
    def __call__(self, img):
        # Drop first 2 channels (VV, VH) - keep channels 2-13 (Sentinel-2 bands)
        return img[2:, :, :]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    DropSARChannels(),  # Drop SAR bands, keep only Sentinel-2 bands
    NormalizeWithStats(train_mean, train_std),  # Standard normalization with S2A stats
])

# --- Country-specific train/val split function ---
meta = pd.read_parquet(datapath["metadata_parquet"])

def load_country_train_val(
    country: str,
    n_samples: int,
    seed: int = None,
    train_frac: float = 0.8,
    img_size=(14, 224, 224),  # Changed from (14, 120, 120) to load all channels
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
ds_train, ds_val = load_country_train_val("Ireland", n_samples=100, seed=seed)

# DataLoaders for train and validation
train_loader = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=4)

# --- Model setup ---
sys.path.append('/home/arne/softcon')
from models.dinov2 import vision_transformer as dinov2_vitb

model_vitb14 = dinov2_vitb.__dict__['vit_base'](
    img_size=224,
    patch_size=14,
    in_chans=12,  #
    block_chunks=0,
    init_values=1e-4,
    num_register_tokens=0,
)

# Load pretrained weights excluding patch embedding layer
ckpt_vitb14 = torch.load('/faststorage/softcon/pretrained/B13_vitb14_softcon_enc.pth', map_location='cpu')

# Exclude patch embedding weights due to channel mismatch (12 vs 13)
model_state = model_vitb14.state_dict()
pretrained_state = {k: v for k, v in ckpt_vitb14.items() 
                   if k not in ['patch_embed.proj.weight', 'patch_embed.proj.bias']}
model_state.update(pretrained_state)
model_vitb14.load_state_dict(model_state)

print("✅ Loaded pretrained weights (excluding patch embedding due to channel mismatch: 12 vs 13)")
print("✅ Patch embedding layer randomly initialized for 12 Sentinel-2 bands")
print(model_vitb14)

# Wrap with LoRA
sys.path.append('/home/arne/LoRA-ViT')
from lora import LoRA_ViT_timm

# Check if the LoRA implementation has unnecessary layers
lora_model = LoRA_ViT_timm(model_vitb14, num_classes = 0 , r=4, alpha=16)

# Remove the unused proj_3d layer if it exists
if hasattr(lora_model, 'proj_3d'):
    delattr(lora_model, 'proj_3d')

# Add classification head
num_classes = 19
classifier = nn.Linear(model_vitb14.embed_dim, num_classes)
lora_with_head = nn.Sequential(lora_model, classifier)

# IMPORTANT: Ensure classification head is trainable
for param in classifier.parameters():
    param.requires_grad = True

print(lora_with_head)
print('\nNumber of trainable parameters:', sum(p.numel() for p in lora_with_head.parameters() if p.requires_grad))

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_with_head.to(device)

# Setup logging
log_file = "/faststorage/softcon/finetuning/training_log.csv"
first_time = True  # Set to False if appending to existing log

# --- Enhanced Training loop with progress tracking ---
epochs = 25
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_with_head.parameters()), lr=1e-4)

with open(log_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    # Write header (only once)
    if first_time:
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    for epoch in range(epochs):
        lora_with_head.train()
        total_loss = 0
        total = 0
        correct = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = lora_with_head(imgs)
            if labels.dtype != torch.float:
                labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy: outputs are logits, apply sigmoid
            pred_probs = torch.sigmoid(outputs)  # shape: [batch_size, num_classes]
            predicted = (pred_probs > 0.5).float()  # shape: [batch_size, num_classes]
            # Accuracy: total correct predictions across all labels
            correct += (predicted == labels).sum().item()
            total += labels.numel()

            # Update progress bar
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_acc = 100.*correct/total
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        lora_with_head.eval()
        val_loss = 0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                if labels.dtype != torch.float:
                    labels = labels.float()
                outputs = lora_with_head(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                pred_probs = torch.sigmoid(outputs)  # shape: [batch_size, num_classes]
                predicted = (pred_probs > 0.5).float()  # shape: [batch_size, num_classes]
                # Accuracy: total correct predictions across all labels
                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.*val_correct/val_total
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log to CSV
        writer.writerow([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])

# Save model
save_path = "/faststorage/softcon/finetuning/ireland_softcon.pth"
torch.save(lora_with_head.state_dict(), save_path)
print(f"Model saved to {save_path}")