import random
import warnings

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score
from torchvision import transforms

# at the top of your script/notebook
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import os
from datetime import datetime

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



from model.video_vit import vit_base_patch8_128
from utils.pos_embed import interpolate_pos_embed

# Use environment variable or relative path


def load_mae_encoder(model, ckpt_path):
    state_dict = model.state_dict()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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


def load_model(r=4, use_lora=True):
    # --- Model setup ---

    # load pretrained weights
    model = vit_base_patch8_128(sep_pos_embed=True, num_classes=19)
    model = load_mae_encoder(model, dir_pretrained)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"ViT trainable parameters w/o LoRA: {num_params}"
    )  # trainable parameters: 86859496

    if use_lora:  # This is the cleanest way
        # Wrap with LoRA
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
    def __init__(self, model, num_classes, lr=1e-4):
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
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


# --- Wrap train_model_replay with Lightning ---
def train_model(lora_model, train_loader, val_loader, epochs=25, lr=1e-4):
    """
    Train the SpectralGPT model using PyTorch Lightning.
    """
    # Wrap the model in a LightningModule
    num_classes = 19  # Update this based on your dataset
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
            size=(128, 128),
            scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),  # Random resized crop
        NormalizeWithStats(train_mean, train_std),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        SelectChannels(),
        NormalizeWithStats(train_mean, train_std),
    ]
)
