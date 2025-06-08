import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import average_precision_score

from configilm import util
from configilm.extra.DataSets import BENv2_DataSet
from tqdm import tqdm

import os
from datetime import datetime
import csv

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
generator = torch.Generator().manual_seed(seed)

util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

os.environ["XFORMERS_DISABLED"] = "1" # for CPU testing


# SOFTCON PRETRAINING band information # Added B10 as a zero-initialized channel
ALL_BANDS_S2_L2A = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]

# Band statistics: mean & std (calculated from 50k data)
S2A_MEAN = [
    752.40087073,
    884.29673756,
    1144.16202635,
    1297.47289228,
    1624.90992062,
    2194.6423161,
    2422.21248945,
    2517.76053101,
    2581.64687018,
    2645.51888987,
    0,
    2368.51236873,
    1805.06846033,
]

S2A_STD = [
    1108.02887453,
    1155.15170768,
    1183.6292542,
    1368.11351514,
    1370.265037,
    1355.55390699,
    1416.51487101,
    1474.78900051,
    1439.3086061,
    1582.28010962,
    1,
    1455.52084939,
    1343.48379601,
]

datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

# Timestamp for logging (similar to SpectralGPT.py)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"softcon_train_log_{timestamp}.csv"
first_time = not os.path.exists(log_file)


def load_softcon_encoder(model, ckpt_path):
    """Load SOFTCON pretrained weights (similar to load_mae_encoder in SpectralGPT.py)"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    msg = model.load_state_dict(ckpt)
    print("Loaded SOFTCON weights with:", msg)
    return model


# --- Model Loading ---
def load_model(r=4):
    """Load SOFTCON model with LoRA and classification head"""
    sys.path.append('/home/arne/softcon')
    from models.dinov2 import vision_transformer as dinov2_vitb

    # Create model
    model_vitb14 = dinov2_vitb.__dict__['vit_base'](
        img_size=224,
        patch_size=14,
        in_chans=13,  # Sentinel-2 bands only
        block_chunks=0,
        init_values=1e-4,
        num_register_tokens=0,
    )

    # Load pretrained weights
    ckpt_vitb14 = torch.load('/faststorage/softcon/pretrained/B13_vitb14_softcon_enc.pth', map_location='cpu', weights_only=True)
    model_state = model_vitb14.state_dict()
    model_vitb14.load_state_dict(model_state)

    print(model_vitb14)

    # Wrap with LoRA
    sys.path.append('/home/arne/LoRA-ViT')
    from lora import LoRA_ViT_timm

    lora_model = LoRA_ViT_timm(model_vitb14, num_classes=0, r=r, alpha=16)

    # Add classification head
    num_classes = 19
    classifier = nn.Linear(model_vitb14.embed_dim, num_classes)
    lora_with_head = nn.Sequential(lora_model, classifier)

    # Ensure classification head is trainable
    for param in classifier.parameters():
        param.requires_grad = True

    print(
        "Number of trainable parameters (w/ LoRA):",
        sum(p.numel() for p in lora_with_head.parameters() if p.requires_grad),
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_with_head.to(device)
    return lora_with_head


def train_model_replay(lora_model, train_loader, val_loader, epochs=25, lr=1e-4):
    """Train SOFTCON model (similar to SpectralGPT.py train_model_replay)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, lora_model.parameters()), lr=lr
    )

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if first_time:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(epochs):
            lora_model.train()
            total_loss = 0
            all_labels = []
            all_scores = []
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

                # Collect predictions and labels for micro AP computation
                pred_probs = torch.sigmoid(outputs)  # Probabilities [0, 1]
                all_scores.append(pred_probs.detach().cpu())
                all_labels.append(labels.detach().cpu())

                # Update progress bar
                loop.set_postfix(loss=loss.item())

            # Compute micro AP for the epoch
            all_scores = torch.cat(all_scores, dim=0).numpy()  # Shape [N, C]
            all_labels = torch.cat(all_labels, dim=0).numpy()  # Shape [N, C]
            micro_ap = average_precision_score(all_labels, all_scores, average='micro')

            avg_train_loss = total_loss / len(train_loader)

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Micro AP: {micro_ap:.4f}"
            )

    return lora_model


def train_model_no_replay():
    """Placeholder for no-replay training (similar to SpectralGPT.py)"""
    pass  # TODO: implement this


def eval_model(lora_model, test_loader):
    """Evaluate SOFTCON model with micro/macro AP and accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)
    lora_model.eval()

    all_labels = []
    all_scores = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device).float()

            # Forward pass
            outputs = lora_model(imgs)              # Raw logits, shape [B, C]
            probs = torch.sigmoid(outputs)          # Probabilities [0, 1]

            # Accumulate for AP computation
            all_scores.append(probs.cpu())
            all_labels.append(labels.cpu())

            # For accuracy
            preds = (probs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.numel()

    # Stack everything
    all_scores = torch.cat(all_scores, dim=0).numpy()  # Shape [N, C]
    all_labels = torch.cat(all_labels, dim=0).numpy()  # Shape [N, C]

    # Average-precision scores
    micro_ap = average_precision_score(all_labels, all_scores, average='micro')
    macro_ap = average_precision_score(all_labels, all_scores, average='macro')

    # Accuracy
    val_acc = 100.0 * val_correct / val_total

    return {
        'micro_ap': micro_ap,
        'macro_ap': macro_ap,
        'accuracy': val_acc
    }


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


class ZeroInitializeB10:
    """Zero-initialize the B10 layer in the input image."""

    def __call__(self, img):
        b10 = torch.zeros((1, img.shape[1], img.shape[2]))  # Shape: [1, H, W]
        img = torch.cat([img[:9, :, :], b10, img[9:, :, :]], dim=0)
        return img


# Use pre-calculated S2A statistics
train_mean, train_std = np.array(S2A_MEAN), np.array(S2A_STD)

# Transformation pipeline for training (with augmentations)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    DropSARChannels(),  # Drop SAR bands, keep only Sentinel-2 bands
    ZeroInitializeB10(),  # Zero-initialize the B10 layer
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.RandomChoice([  # Randomly apply one of the rotations
        transforms.RandomRotation(degrees=(0, 0)),  # No rotation
        transforms.RandomRotation(degrees=(90, 90)),  # Rotate 90 degrees
        transforms.RandomRotation(degrees=(180, 180)),  # Rotate 180 degrees
        transforms.RandomRotation(degrees=(270, 270)),  # Rotate 270 degrees
    ]),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random resized crop
    NormalizeWithStats(train_mean, train_std),  # Standard normalization with S2A stats
])

# Transformation pipeline for validation (no augmentations)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    DropSARChannels(),  # Drop SAR bands, keep only Sentinel-2 bands
    ZeroInitializeB10(),  # Zero-initialize the B10 layer
    NormalizeWithStats(train_mean, train_std),  # Standard normalization with S2A stats
])

# --- Model setup ---
sys.path.append('/home/arne/softcon')
from models.dinov2 import vision_transformer as dinov2_vitb

model_vitb14 = dinov2_vitb.__dict__['vit_base'](
    img_size=224,
    patch_size=14,
    in_chans=13,  #
    block_chunks=0,
    init_values=1e-4,
    num_register_tokens=0,
)
