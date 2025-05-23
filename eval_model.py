import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, average_precision_score

from configilm import util
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule


datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

# Load saved predictions and labels
all_preds = np.load("/faststorage/softcon/finetuning/baseline_preds.npy")
all_labels = np.load("/faststorage/softcon/finetuning/baseline_labels.npy")

# Binarize predictions at threshold 0.5 for multilabel metrics
preds_bin = (all_preds > 0.5).astype(np.int32)

# Compute metrics
f1 = f1_score(all_labels, preds_bin, average='macro')
acc = accuracy_score(all_labels, preds_bin)
map_score = average_precision_score(all_labels, all_preds, average='macro')

print(all_preds)


print(f"Test set macro F1: {f1:.4f}")
print(f"Test set macro accuracy: {acc:.4f}")
print(f"Test set macro mAP: {map_score:.4f}")





