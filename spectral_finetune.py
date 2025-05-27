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
from tqdm import tqdm

import os

from datetime import datetime
import os
import csv

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = f'train_log_{timestamp}.csv'
first_time = not os.path.exists(log_file)

dir_pretrained = "/faststorage/SpectralGPT/SpectralGPT.pth"
dir_pretrained_plus = "/faststorage/SpectralGPT/S+/SpectralGPT+.pth"
dir_output = "./weights/"
include_snowy = False  # Include patches with snow
include_cloudy = False  # Include patches with clouds

# Set random seeds for reproducibility
seed = 42
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


sys.path.append('/faststorage/shuocheng/LoRA_ViT')
from pos_embed import interpolate_pos_embed
def load_mae_encoder(model, ckpt_path, headless=True):
    state_dict = model.state_dict()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_model = ckpt.get('model', ckpt)
    # remove decoder & mask
    ckpt_model = {k: v for k, v in ckpt_model.items()
             if not (k.startswith('decoder') or k.startswith('mask_token'))}

    # Delete mismatch (inherited from SpectralGPT finetuning)
    # Delete head (embed_size, 10) for downstream 19-class classification
    for k in ['patch_embed.0.proj.weight', 'patch_embed.1.proj.weight', 'patch_embed.2.proj.weight',
            'patch_embed.2.proj.bias', 'head.weight', 'head.bias']:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del ckpt_model[k]
    
    # pos_embed interpolation
    if 'pos_embed_spatial' in ckpt_model:
        interpolate_pos_embed(model, ckpt_model)

    # strict=False to ignore missing head.weight/bias 
    msg = model.load_state_dict(ckpt_model, strict=False)
    print('Loaded with:', msg)
    # msg.missing_keys:     head.weight/bias；
    # msg.unexpected_keys:  empty
    return model


util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

# --- Normalization function as provided ---
class SelectChannels:
    def __call__(self, img, channels: list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]):
        # img is a torch.Tensor [C, H, W]
        return img[channels, :, :]

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
    transforms.Resize((128, 128)),
SelectChannels(),

])

# Load dataset with temp transform for stats
full_dataset = BENv2_DataSet.BENv2DataSet(
    data_dirs=datapath,
    transform=temp_transform,
    img_size=(14, 128, 128),
    include_snowy=include_snowy,
    include_cloudy=include_cloudy
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
    transforms.Resize((128, 128)),
    SelectChannels(),
    NormalizeWithStats(train_mean, train_std),
])

# --- Country-specific train/val split function ---
meta = pd.read_parquet(datapath["metadata_parquet"])

def load_country_train_val(
    country: str,
    n_samples: int,
    seed: int = None,
    train_frac: float = 0.8,
    img_size=(14, 128, 128),
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
ds_train, ds_val = load_country_train_val("Ireland", n_samples=4874, seed=seed, include_snowy=include_snowy, include_cloudy=include_cloudy)


# DataLoaders for train and validation
train_loader = DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=4)

# --- Model setup ---
sys.path.append('/faststorage/shuocheng/LoRA_ViT')
from LoRA_ViT.video_vit import vit_base_patch8_128

# load pretrained weights
num_classes = 19
model = vit_base_patch8_128(sep_pos_embed=True, num_classes = 19)
model = load_mae_encoder(model, dir_pretrained)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ViT trainable parameters w/o LoRA: {num_params}") #trainable parameters: 86859496

# Wrap with LoRA
sys.path.append('/home/arne/LoRA-ViT')
from LoRA_ViT.lora import LoRA_SViT

lora_model = LoRA_SViT(model, r=4, alpha=16)
print(lora_model)
print('\nNumber of trainable parameters: (w/ LoRA)', sum(p.numel() for p in lora_model.parameters() if p.requires_grad))

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model.to(device)

# --- Training loop ---
epochs = 25
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=1e-4)

with open(log_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    # 写表头（只写一次）
    if first_time:
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    for epoch in range(epochs):
        lora_model.train()
        total_loss = 0
        total = 0
        correct = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
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

            # 预测时：outputs 是 logits, 应该用 sigmoid
            pred_probs = torch.sigmoid(outputs)  # shape: [batch_size, num_classes]
            predicted = (pred_probs > 0.5).float()  # shape: [batch_size, num_classes]
            # Accuracy: 可以用 total correct predictions
            correct += (predicted == labels).sum().item()
            total += labels.numel()

            # Update tqdm bar
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        train_acc = 100.*correct/total
        avg_train_loss = total_loss / len(train_loader)

        # Validation loss
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

                pred_probs = torch.sigmoid(outputs)  # shape: [batch_size, num_classes]
                predicted = (pred_probs > 0.5).float()  # shape: [batch_size, num_classes]
                # Accuracy: 可以用 total correct predictions
                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.*val_correct/val_total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        writer.writerow([epoch, avg_train_loss, 100.*correct/total, avg_val_loss, val_acc])

save_path = "/faststorage/SpectralGPT/finetuning/finland.pth"
torch.save(lora_model.state_dict(), save_path)
print(f"Model saved to {save_path}")