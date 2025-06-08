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
from sklearn.metrics import average_precision_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# at the top of your script/notebook
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import os

from datetime import datetime
import os
import csv

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"train_log_{timestamp}.csv"
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


sys.path.append("/faststorage/shuocheng/LoRA_SpectralGPT")
from pos_embed import interpolate_pos_embed
from LoRA_ViT.video_vit import vit_base_patch8_128
# Use environment variable or relative path


def load_mae_encoder(model, ckpt_path):
    state_dict = model.state_dict()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_model = ckpt.get("model", ckpt)
    # remove decoder & mask
    ckpt_model = {
        k: v
        for k, v in ckpt_model.items()
        if not (k.startswith("decoder") or k.startswith("mask_token"))
    }

    # Delete mismatch (inherited from MAE)
    for k in [
        "patch_embed.0.proj.weight",
        "patch_embed.1.proj.weight",
        "patch_embed.2.proj.weight",
        "patch_embed.2.proj.bias",
        "head.weight",
        "head.bias",
    ]:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del ckpt_model[k]
    # Delete head (embed_size, 10) for downstream 19-class classification
    # pos_embed interpolation
    if "pos_embed_spatial" in ckpt_model:
        interpolate_pos_embed(model, ckpt_model)

    # strict=False to ignore missing head.weight/bias
    msg = model.load_state_dict(ckpt_model, strict=False)
    print("Loaded with:", msg)
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
    print(
        f"ViT trainable parameters w/o LoRA: {num_params}"
    )  # trainable parameters: 86859496

    # Wrap with LoRA
    sys.path.append("/home/arne/LoRA-ViT")
    from LoRA_ViT.lora import LoRA_SViT

    lora_model = LoRA_SViT(model, r=r, alpha=16)

    print(lora_model)
    print(
        "\nNumber of trainable parameters: (w/ LoRA)",
        sum(p.numel() for p in lora_model.parameters() if p.requires_grad),
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)

    return lora_model


def train_model_replay(lora_model, train_loader, val_loader, epochs=25, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)

    # --- Training loop ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, lora_model.parameters()), lr=lr
    )

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if first_time:
            writer.writerow(
                ["epoch", "train_loss", "train_micro_ap", "val_loss", "val_micro_ap"]
            )

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

                # Update tqdm bar
                loop.set_postfix(loss=f"{loss.item():.4f}")

            # Compute micro AP for training
            all_scores = torch.cat(all_scores, dim=0).numpy()  # Shape [N, C]
            all_labels = torch.cat(all_labels, dim=0).numpy()  # Shape [N, C]
            train_micro_ap = average_precision_score(
                all_labels, all_scores, average="micro"
            )
            avg_train_loss = total_loss / len(train_loader)

            # Validation loop
            lora_model.eval()
            val_loss = 0
            val_labels = []
            val_scores = []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    if labels.dtype != torch.float:
                        labels = labels.float()
                    outputs = lora_model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Collect predictions and labels for micro AP computation
                    pred_probs = torch.sigmoid(outputs)  # Probabilities [0, 1]
                    val_scores.append(pred_probs.cpu())
                    val_labels.append(labels.cpu())

            # Compute micro AP for validation
            val_scores = torch.cat(val_scores, dim=0).numpy()  # Shape [N, C]
            val_labels = torch.cat(val_labels, dim=0).numpy()  # Shape [N, C]
            val_micro_ap = average_precision_score(
                val_labels, val_scores, average="micro"
            )
            avg_val_loss = val_loss / len(val_loader)

            # Log results
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Micro AP: {train_micro_ap:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Micro AP: {val_micro_ap:.4f}"
            )
            writer.writerow(
                [epoch + 1, avg_train_loss, train_micro_ap, avg_val_loss, val_micro_ap]
            )

    return lora_model


def train_model_no_replay():
    pass  # NOTE we can use the replay none


def eval_model(lora_model, test_loader):
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

            # forward
            outputs = lora_model(imgs)  # raw logits, shape [B, C]
            probs = torch.sigmoid(outputs)  # probabilities [0,1]

            # accumulate for AP computation
            all_scores.append(probs.cpu())
            all_labels.append(labels.cpu())

            # for accuracy
            preds = (probs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.numel()

    # stack everything
    all_scores = torch.cat(all_scores, dim=0).numpy()  # shape [N, C]
    all_labels = torch.cat(all_labels, dim=0).numpy()  # shape [N, C]

    # average-precision scores
    micro_ap = average_precision_score(all_labels, all_scores, average="micro")
    macro_ap = average_precision_score(all_labels, all_scores, average="macro")

    # accuracy
    val_acc = 100.0 * val_correct / val_total

    return {"micro_ap": micro_ap, "macro_ap": macro_ap, "accuracy": val_acc}


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


class SelectChannels:
    def __call__(self, img):
        # img is a torch.Tensor [C, H, W]
        return img[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], :, :]


train_mean = [
    1370.19151926,
    1184.3824625,
    1120.77120066,
    1136.26026392,
    1263.73947144,
    1645.40315151,
    1846.87040806,
    1762.59530783,
    1972.62420416,
    582.72633433,
    1732.16362238,
    1247.91870117,
]
train_std = [
    633.15169573,
    650.2842772,
    712.12507725,
    965.23119807,
    948.9819932,
    1108.06650639,
    1258.36394548,
    1233.1492281,
    1364.38688993,
    472.37967789,
    1310.36996126,
    1087.6020813,
]

train_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        SelectChannels(),
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomVerticalFlip(),  # Random vertical flip
        transforms.RandomChoice(
            [  # Randomly apply one of the rotations
                transforms.RandomRotation(degrees=(0, 0)),  # No rotation
                transforms.RandomRotation(degrees=(90, 90)),  # Rotate 90 degrees
                transforms.RandomRotation(degrees=(180, 180)),  # Rotate 180 degrees
                transforms.RandomRotation(degrees=(270, 270)),  # Rotate 270 degrees
            ]
        ),
        transforms.RandomResizedCrop(
            size=(128, 128), scale=(0.8, 1.0)
        ),  # Random resized crop
        NormalizeWithStats(
            train_mean, train_std
        ), 
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        SelectChannels(),
        NormalizeWithStats(
            train_mean, train_std
        ), 
    ]
)
