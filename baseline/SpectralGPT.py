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


import importlib.util
from pathlib import Path


sys.path.append('/faststorage/shuocheng/LoRA_SpectralGPT')
from pos_embed import interpolate_pos_embed
from LoRA_ViT.video_vit import vit_base_patch8_128
# Use environment variable or relative path



def load_mae_encoder(model, ckpt_path):
    state_dict = model.state_dict()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_model = ckpt.get('model', ckpt)
    # remove decoder & mask
    ckpt_model = {k: v for k, v in ckpt_model.items()
             if not (k.startswith('decoder') or k.startswith('mask_token'))}

    # Delete mismatch (inherited from MAE)
    for k in ['patch_embed.0.proj.weight', 'patch_embed.1.proj.weight', 'patch_embed.2.proj.weight',
            'patch_embed.2.proj.bias', 'head.weight', 'head.bias']:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del ckpt_model[k]
    # Delete head (embed_size, 10) for downstream 19-class classification
    # pos_embed interpolation
    if 'pos_embed_spatial' in ckpt_model:
        interpolate_pos_embed(model, ckpt_model)

    # strict=False to ignore missing head.weight/bias
    msg = model.load_state_dict(ckpt_model, strict=False)
    print('Loaded with:', msg)
    # msg.missing_keys:     head.weight/bias；
    # msg.unexpected_keys:  empty
    return model

def load_model(r=4):
    # --- Model setup ---


    # load pretrained weights
    num_classes = 19
    model = vit_base_patch8_128(sep_pos_embed=True, num_classes=19)
    model = load_mae_encoder(model, dir_pretrained)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT trainable parameters w/o LoRA: {num_params}")  # trainable parameters: 86859496

    # Wrap with LoRA
    sys.path.append('/home/arne/LoRA-ViT')
    from LoRA_ViT.lora import LoRA_SViT

    lora_model = LoRA_SViT(model, r=r, alpha=16)

    print(lora_model)
    print('\nNumber of trainable parameters: (w/ LoRA)',
          sum(p.numel() for p in lora_model.parameters() if p.requires_grad))

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)

    return lora_model

def train_model_replay(lora_model,train_loader,val_loader,epochs=25):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)
    # --- Training loop ---
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

                # 预测时：outputs 是 logits, 应该用 sigmoid
                pred_probs = torch.sigmoid(outputs)  # shape: [batch_size, num_classes]
                predicted = (pred_probs > 0.5).float()  # shape: [batch_size, num_classes]
                # Accuracy: 可以用 total correct predictions
                correct += (predicted == labels).sum().item()
                total += labels.numel()

                # Update tqdm bar
                loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
            train_acc = 100. * correct / total
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
            val_acc = 100. * val_correct / val_total
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            writer.writerow([epoch, avg_train_loss, 100. * correct / total, avg_val_loss, val_acc])

    return lora_model


def train_model_no_replay():
    pass #NOTE we can use the replay none

def eval_model(lora_model,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Validation loss
    lora_model.eval()
    val_loss = 0
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
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
    avg_val_loss = val_loss / len(test_loader)
    val_acc = 100. * val_correct / val_total

    return {'accuracy': val_acc}

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

#train_mean, train_std = compute_mean_std(train_dataset)
#print("Train mean:", train_mean)
#print("Train std:", train_std)

train_mean = np.array([344.37332, 419.62582, 599.5149, 577.9469,
                       940.35547, 1804.2218, 2096.4666, 2252.1025,
                       2296.7737, 2302.6838, 1628.6593, 1028.488],
                      dtype=np.float32)
train_std  = np.array([430.6854, 470.55728, 503.29114, 598.5436,
                       658.99994, 1042.0695, 1221.3387, 1316.3477,
                       1301.1089, 1258.8138, 1045.6333,  798.5746],
                      dtype=np.float32)
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

