import sys
import random
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

BENv2_DataSet.BENv2DataSet.get_available_channel_configurations()

class SelectFirst2Channels:
    def __call__(self, img):
        # img is a torch.Tensor [C, H, W]
        return img[:2, :, :]

# --- Normalization function as provided ---
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

# Load dataset with temp transform
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
        img_np = img_np.astype(np.float32) / 255.0  # <-- convert to float32 and scale
        return torch.from_numpy(img_np)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    SelectFirst2Channels(),
    NormalizeWithStats(train_mean, train_std),
])

# Reload datasets with normalization
full_dataset = BENv2_DataSet.BENv2DataSet(
    data_dirs=datapath,
    max_len=1000,
    transform=transform
)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")



######## CHANGE WEIGHTS PATH HERE ########
# Load SoftCon ViT model
sys.path.append('/home/arne/softcon')
from models.dinov2 import vision_transformer as dinov2_vits
#########################################

model_vits14 = dinov2_vits.__dict__['vit_small'](
    img_size=224,
    patch_size=14,
    in_chans=2,
    block_chunks=0,
    init_values=1e-5,
    num_register_tokens=0,
)

# load pretrained weights
# ckpt_vits14 = torch.load('/faststorage/softcon/pretrained/B2_vits14_softcon.pth', map_location='cpu')
model_vits14.load_state_dict(ckpt_vits14)

# Optionally, wrap with LoRA if you want to use it
sys.path.append('/home/arne/LoRA-ViT')
from lora import LoRA_ViT_timm
lora_model = LoRA_ViT_timm(model_vits14, r=4, alpha=16)

# Add classification head
num_classes = 19
classifier = nn.Linear(model_vits14.embed_dim, num_classes)
lora_with_head = nn.Sequential(lora_model, classifier)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_with_head = lora_with_head.to(device)
lora_with_head.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in DataLoader(full_dataset, batch_size=16, shuffle=False, num_workers=4):
        imgs = imgs.to(device)
        outputs = lora_with_head(imgs)
        preds = torch.sigmoid(outputs).cpu().numpy()  # For multilabel, get probabilities
        all_preds.append(preds)
        all_labels.append(labels.numpy())

import numpy as np
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

np.save("/faststorage/softcon/finetuning/baseline_preds.npy", all_preds)
np.save("/faststorage/softcon/finetuning/baseline_labels.npy", all_labels)

print("Predictions shape:", all_preds.shape)
print("Labels shape:", all_labels.shape)