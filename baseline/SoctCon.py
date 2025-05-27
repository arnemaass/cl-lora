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
from tqdm import tqdm

import os
from datetime import datetime
import csv

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

# Timestamp for logging (similar to SpectralGPT.py)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = f'softcon_train_log_{timestamp}.csv'
first_time = not os.path.exists(log_file)

def load_softcon_encoder(model, ckpt_path):
    """Load SOFTCON pretrained weights (similar to load_mae_encoder in SpectralGPT.py)"""
    state_dict = model.state_dict()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Exclude patch embedding weights due to channel mismatch (12 vs 13)
    ckpt_model = {k: v for k, v in ckpt.items() 
                  if k not in ['patch_embed.proj.weight', 'patch_embed.proj.bias']}
    
    # Load with strict=False to handle missing patch embedding
    msg = model.load_state_dict(ckpt_model, strict=False)
    print('Loaded SOFTCON weights with:', msg)
    return model

def load_model(r=4):
    """Load SOFTCON model with LoRA (similar to SpectralGPT.py load_model)"""
    sys.path.append('/home/arne/softcon')
    from models.dinov2 import vision_transformer as dinov2_vitb

    # Create model
    model_vitb14 = dinov2_vitb.__dict__['vit_base'](
        img_size=224,
        patch_size=14,
        in_chans=12,  # Sentinel-2 bands only
        block_chunks=0,
        init_values=1e-4,
        num_register_tokens=0,
    )

    # Load SOFTCON pretrained weights
    model_vitb14 = load_softcon_encoder(model_vitb14, '/faststorage/softcon/pretrained/B13_vitb14_softcon_enc.pth')

    num_params = sum(p.numel() for p in model_vitb14.parameters() if p.requires_grad)
    print(f"SOFTCON ViT trainable parameters w/o LoRA: {num_params}")

    # Wrap with LoRA
    sys.path.append('/home/arne/LoRA-ViT')
    from lora import LoRA_ViT_timm

    lora_model = LoRA_ViT_timm(model_vitb14, num_classes=0, r=r, alpha=16)

    # Remove unused proj_3d layer if it exists
    if hasattr(lora_model, 'proj_3d'):
        delattr(lora_model, 'proj_3d')

    # Add classification head
    num_classes = 19
    classifier = nn.Linear(model_vitb14.embed_dim, num_classes)
    lora_with_head = nn.Sequential(lora_model, classifier)

    # Ensure classification head is trainable
    for param in classifier.parameters():
        param.requires_grad = True

    print('Number of trainable parameters (w/ LoRA):', 
          sum(p.numel() for p in lora_with_head.parameters() if p.requires_grad))

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_with_head.to(device)
    return lora_with_head

def train_model_replay(lora_model, train_loader, val_loader, epochs=25):
    """Train SOFTCON model (similar to SpectralGPT.py train_model_replay)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=1e-4)

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if first_time:
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        
        for epoch in range(epochs):
            lora_model.train()
            total_loss = 0
            total = 0
            correct = 0
            loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=True)
            
            for imgs, labels in loop:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = lora_model(imgs)
                if labels.dtype != torch.float:
                    labels = labels.float()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate accuracy
                pred_probs = torch.sigmoid(outputs)
                predicted = (pred_probs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.numel()

                # Update progress bar
                loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
            
            train_acc = 100. * correct / total
            avg_train_loss = total_loss / len(train_loader)

            # Validation
            lora_model.eval()
            val_loss = 0
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    if labels.dtype != torch.float:
                        labels = labels.float()
                    outputs = lora_model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    pred_probs = torch.sigmoid(outputs)
                    predicted = (pred_probs > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.numel()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            writer.writerow([epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc])

    return lora_model

def train_model_no_replay():
    """Placeholder for no-replay training (similar to SpectralGPT.py)"""
    pass  # TODO: implement this

def eval_model(lora_model, test_loader):
    """Evaluate SOFTCON model (similar to SpectralGPT.py eval_model)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)
    
    lora_model.eval()
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if labels.dtype != torch.float:
                labels = labels.float()
            outputs = lora_model(imgs)

            pred_probs = torch.sigmoid(outputs)
            predicted = (pred_probs > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.numel()
    
    val_acc = 100. * val_correct / val_total
    return {'accuracy': val_acc}

# --- Data preprocessing functions ---
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

class DropSARChannels:
    """Drop SAR channels to keep only Sentinel-2 bands"""
    def __call__(self, img):
        # Drop first 2 channels (VV, VH) - keep channels 2-13 (Sentinel-2 bands)
        return img[2:, :, :]

# Use pre-calculated S2A statistics
train_mean, train_std = np.array(S2A_MEAN), np.array(S2A_STD)

# SOFTCON transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    DropSARChannels(),  # Drop SAR bands, keep only Sentinel-2 bands
    NormalizeWithStats(train_mean, train_std),  # Standard normalization with S2A stats
])

def load_country_train_val(
    country: str,
    n_samples: int,
    seed: int = None,
    train_frac: float = 0.8,
    img_size=(14, 224, 224),
    include_snowy=False,
    include_cloudy=False,
    split="train"
):
    """Load country-specific data (similar to SpectralGPT.py but with SOFTCON transforms)"""
    meta = pd.read_parquet(datapath["metadata_parquet"])
    
    mask = (meta.country == country) & (meta.split == split)
    available = meta.loc[mask, "patch_id"].tolist()
    if not available:
        raise ValueError(f"No {split} patches found for country={country!r}")
    if n_samples > len(available):
        print(f"Warning: Requested {n_samples} samples but only {len(available)} available. Using all.")
        n_samples = len(available)
    
    rng = random.Random(seed)
    sampled = rng.sample(available, k=n_samples)
    
    # Split for train/val like baseline pattern
    split_at = int(n_samples * train_frac)
    train_ids = set(sampled[:split_at])
    val_ids = set(sampled[split_at:])
    ids = set(sampled[:])  # All samples
    
    def _make_ds(keep_ids):
        return BENv2_DataSet.BENv2DataSet(
            data_dirs=datapath,
            img_size=img_size,
            split=split,
            include_snowy=include_snowy,
            include_cloudy=include_cloudy,
            patch_prefilter=lambda pid: pid in keep_ids,
            transform=transform  # Use SOFTCON transforms
        )
    
    # Return datasets like baseline pattern
    train_ds = _make_ds(train_ids)
    val_ds = _make_ds(val_ids)
    return train_ds, val_ds