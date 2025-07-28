"""
SpectralGPT.py
SpectralGPT model implementation for remote sensing image classification.

This module provides:
1. SpectralGPT model loading and initialization with LoRA support
2. PyTorch Lightning training framework
3. Model evaluation with multi-label classification metrics
4. Data preprocessing and augmentation transforms
5. Pre-trained weight loading and fine-tuning capabilities

The module supports:
- Vision Transformer (ViT) backbone with LoRA adaptation
- Multi-label classification (19 classes)
- Continual learning with replay mechanisms
- Comprehensive evaluation metrics (accuracy, micro/macro AP)

Dependencies
------------
• torch, pytorch_lightning, torchvision
• numpy, sklearn, random
• Custom model modules (video_vit, lora_vit)
"""

import random
import warnings

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score
from torchvision import transforms
from typing import Dict, Any

# Suppress sklearn warnings for undefined metrics
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import os
from datetime import datetime

# Setup logging and file paths
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"train_log_{timestamp}.csv"
first_time = not os.path.exists(log_file)

# Pre-trained model paths
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


from model.video_vit import vit_base_patch8_128
from utils.pos_embed import interpolate_pos_embed


def load_mae_encoder(model, ckpt_path):
    """
    Load pre-trained MAE encoder weights into the model.
    
    This function handles loading pre-trained weights from a MAE (Masked Autoencoder)
    checkpoint, removing decoder components and handling shape mismatches.
    
    Args:
        model: The target model to load weights into
        ckpt_path: Path to the pre-trained checkpoint file
    
    Returns:
        model: Model with loaded pre-trained weights
    """
    state_dict = model.state_dict()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_model = ckpt.get("model", ckpt)
    
    # Remove decoder & mask components (keep only encoder)
    ckpt_model = {
        k: v
        for k, v in ckpt_model.items()
        if not (k.startswith("decoder") or k.startswith("mask_token"))
    }

    # Delete mismatched keys inherited from MAE
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
    
    # Delete head for downstream 19-class classification
    # Interpolate positional embeddings if needed
    if "pos_embed_spatial" in ckpt_model:
        interpolate_pos_embed(model, ckpt_model)

    # Load state dict with strict=False to ignore missing head.weight/bias
    msg = model.load_state_dict(ckpt_model, strict=False)
    print("Loaded with:", msg)
    # msg.missing_keys:     head.weight/bias
    # msg.unexpected_keys:  empty
    return model


def load_model(r=4, use_lora: bool = True):
    """
    Load the SpectralGPT model with optional LoRA adaptation.
    
    This function initializes a Vision Transformer model, loads pre-trained weights,
    and optionally wraps it with LoRA for efficient fine-tuning.
    
    Args:
        r: LoRA rank (reduction factor for low-rank adaptation)
        use_lora: Whether to apply LoRA adaptation to the model
    
    Returns:
        model: Loaded model (with or without LoRA)
    """
    # --- Model setup ---
    # Initialize base ViT model
    model = vit_base_patch8_128(sep_pos_embed=True, num_classes=19)
    model = load_mae_encoder(model, dir_pretrained)

    # Print trainable parameters count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"ViT trainable parameters w/o LoRA: {num_params}"
    )  # trainable parameters: 86859496

    if use_lora:  # Wrap with LoRA for efficient adaptation
        from model.lora_vit import LoRA_SViT

        lora_model = LoRA_SViT(model, r=r, alpha=16, num_classes=19)

        print(lora_model)
        print(
            "\nNumber of trainable parameters: (w/ LoRA)",
            sum(p.numel() for p in lora_model.parameters() if p.requires_grad),
        )

        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lora_model.to(device)

        return lora_model
    else:
        # Move base model to device as well
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model


# --- Lightning Module --- 
class SpectralGPTLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for SpectralGPT training.
    
    This class provides a complete training framework with:
    - Forward pass implementation
    - Training and validation steps
    - Loss computation and metrics logging
    - Optimizer and learning rate scheduler configuration
    """
    
    def __init__(self, model, num_classes, lr=1e-4):
        """
        Initialize the Lightning module.
        
        Args:
            model: The SpectralGPT model
            num_classes: Number of output classes (19 for multi-label)
            lr: Learning rate for optimization
        """
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for multi-label
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step implementation.
        
        Args:
            batch: Input batch (images, labels)
            batch_idx: Batch index
        
        Returns:
            loss: Training loss
        """
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step implementation.
        
        Args:
            batch: Input batch (images, labels)
            batch_idx: Batch index
        """
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            dict: Optimizer and scheduler configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # Minimize the validation loss
            factor=0.1,  # Reduce LR by a factor of 10
            patience=2,  # Wait for epochs without improvement
            threshold=0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor
            },
        }


# --- Training function ---
def train_model(lora_model, train_loader, val_loader, epochs=25, lr=1e-4):
    """
    Train the SpectralGPT model using PyTorch Lightning.
    
    This function provides a complete training pipeline with:
    - Lightning module wrapping
    - Automatic GPU/CPU detection
    - Model checkpointing and logging
    - Training progress monitoring
    
    Args:
        lora_model: The LoRA-adapted SpectralGPT model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        pl_model: Trained PyTorch Lightning model
    """
    # Wrap the model in a LightningModule
    num_classes = 19  # Multi-label classification
    pl_model = SpectralGPTLightningModule(lora_model, num_classes=num_classes, lr=lr)

    # Define PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",  # Automatically use GPU if available
        log_every_n_steps=10,
        default_root_dir="./",  # Change this to your desired save directory
    )

    # Train the model
    trainer.fit(pl_model, train_loader, val_loader)

    # Save the trained model
    save_path = "./trained_model.pt"
    torch.save(pl_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return pl_model


def eval_model(lora_model, test_loader):
    """
    Evaluate the SpectralGPT model on test data.
    
    This function computes comprehensive evaluation metrics for multi-label
    classification including accuracy, micro-average precision, and macro-average precision.
    
    Args:
        lora_model: The trained SpectralGPT model
        test_loader: Test data loader
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
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
            outputs = lora_model(imgs)  # raw logits, shape [B, C]
            probs = torch.sigmoid(outputs)  # probabilities [0,1]

            # Accumulate for AP computation
            all_scores.append(probs.cpu())
            all_labels.append(labels.cpu())

            # Calculate accuracy
            preds = (probs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.numel()

    # Stack all predictions and labels
    all_scores = torch.cat(all_scores, dim=0).numpy()  # shape [N, C]
    all_labels = torch.cat(all_labels, dim=0).numpy()  # shape [N, C]

    # Calculate average precision scores
    micro_ap = average_precision_score(all_labels, all_scores, average="micro")
    macro_ap = average_precision_score(all_labels, all_scores, average="macro")

    # Calculate overall accuracy
    val_acc = 100.0 * val_correct / val_total

    return {"micro_ap": micro_ap, "macro_ap": macro_ap, "accuracy": val_acc}


# --- Data preprocessing functions ---
class NormalizeWithStats:
    """
    Custom normalization transform using SOFTCON statistics.
    
    This transform applies standard normalization using pre-computed
    mean and standard deviation values for S2A (Sentinel-2) data.
    """

    def __init__(self, mean, std):
        """
        Initialize normalization transform.
        
        Args:
            mean: Mean values for each channel
            std: Standard deviation values for each channel
        """
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, img):
        """
        Apply normalization to input image.
        
        Args:
            img: Input image tensor
        
        Returns:
            torch.Tensor: Normalized image tensor
        """
        img_np = img.numpy().astype(np.float32)
        # Standard normalization: (x - mean) / std
        img_np = (img_np - self.mean) / self.std
        return torch.from_numpy(img_np).float()


class SelectChannels:
    """
    Channel selection transform for Sentinel-2 data.
    
    This transform selects specific channels from the input image,
    typically used to focus on relevant spectral bands for classification.
    """
    
    def __call__(self, img):
        """
        Select specific channels from input image.
        
        Args:
            img: Input image tensor [C, H, W]
        
        Returns:
            torch.Tensor: Image with selected channels
        """
        # Select channels 2-13 (specific spectral bands) to align the channels of SpectralGPT and BENv2
        return img[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], :, :]


# Pre-computed normalization statistics for S2A data
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

# Training data augmentation pipeline
train_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Resize to standard size
        SelectChannels(),  # Select relevant spectral bands
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
            size=(128, 128),
            scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),  # Random resized crop for additional augmentation
        NormalizeWithStats(train_mean, train_std),  # Apply normalization
    ]
)

# Validation data preprocessing pipeline (no augmentation)
val_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Resize to standard size
        SelectChannels(),  # Select relevant spectral bands
        NormalizeWithStats(train_mean, train_std),  # Apply normalization
    ]
)
