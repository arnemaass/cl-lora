"""
baseline.py
Baseline implementation for continual learning experiments with remote sensing data.

This module provides baseline implementations for different continual learning strategies:
1. Fine-tuning from scratch with full replay
2. Continual fine-tuning without replay
3. Task-specific tuning for LoRA adapter generation

The module supports:
- BigEarthNet-V2 dataset loading and preprocessing
- Multiple model architectures (SpectralGPT, SoftCon)
- PyTorch Lightning training framework
- Comprehensive evaluation and logging
- LoRA parameter saving and loading

Key Features:
- Dynamic dataset loading based on model architecture
- Configurable training strategies
- Multi-label classification evaluation
- Automatic checkpointing and model saving

Dependencies
------------
• torch, pytorch_lightning, pandas, joblib
• configilm, sklearn, yaml
• Custom model modules (SpectralGPT, SoftCon)
"""

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import pytorch_lightning as L
import torch
import yaml

# --- DataSet factory based on configilm / BENv2 ---
from configilm import util
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.metrics import average_precision_score
from torch.utils.data import ConcatDataset, DataLoader

# Data paths for BigEarthNet-V2 dataset
datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}
util.MESSAGE_LEVEL = util.MessageLevel.INFO

# Disable xFormers for debugging on CPU
if not torch.cuda.is_available():
    os.environ["XFORMERS_DISABLED"] = "1"


# -- Trainer Parameters ---
def create_trainer(params: Dict[str, Any]) -> L.Trainer:
    """
    Create and configure a central PyTorch Lightning Trainer.
    
    This function sets up a trainer with early stopping, checkpointing,
    and learning rate scheduling for consistent training across experiments.
    
    Args:
        params: Configuration dictionary containing training parameters
    
    Returns:
        L.Trainer: Configured PyTorch Lightning trainer
    """
    # Model checkpoint callback for saving best models
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        # dirpath=params.get("save_dir", "./saved_models"),  # Directory to save checkpoints
        # filename="{country}",  # Placeholder for dynamic naming        save_top_k=1,  # Save only the best model
        mode="min",  # Minimize the validation loss
        verbose=True,
    )

    # Learning rate monitor callback for tracking LR changes
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Define the trainer with distributed training support
    trainer = L.Trainer(
        max_epochs=params.get("epoch", 15),
        accelerator="auto",  # Automatically use GPU if available
        log_every_n_steps=50,
        strategy=L.strategies.DDPStrategy(
            find_unused_parameters=True
        ),  # Allow unused parameters for LoRA
        default_root_dir=params.get(
            "save_dir", "./saved_models"
        ),  # Directory for logs and checkpoints
        callbacks=[checkpoint_callback, lr_monitor],  # Add callbacks
    )
    return trainer


# --- Data Loading ---
def load_country_train_val(
    country: str,
    n_samples: int,
    seed: int = None,
    include_snowy=False,
    include_cloudy=False,
    split="train",
    frac=0.8,
    model_module=None,  # Specify the model module
):
    """
    Load country-specific train/validation datasets.
    
    This function dynamically adjusts transformations and image size based on the
    model architecture to ensure compatibility between different model types.
    
    Args:
        country: Country name to load data for
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        include_snowy: Whether to include snowy patches
        include_cloudy: Whether to include cloudy patches
        split: Data split ('train' or 'test')
        frac: Fraction for train/val split
        model_module: Model architecture ('SpectralGPT' or 'SoftCon')
    
    Returns:
        tuple: (train_dataset, val_dataset) for the specified country
    """
    # Dynamically adjust transformations and img_size based on model architecture
    transform = train_transform if split == "train" else val_transform
    if model_module == "SoftCon":
        img_size_ben = (14, 224, 224)  # SoftCon-specific img_size for BENv2 loader
    elif model_module == "SpectralGPT":
        img_size_ben = (14, 128, 128)  # SpectralGPT-specific img_size for BENv2 loader
    else:
        raise ValueError(f"Unknown model_module: {model_module}")

    # Load metadata and filter by country and split
    meta = pd.read_parquet(datapath["metadata_parquet"])

    # Filter metadata based on the split
    mask = (meta.country == country) & (meta.split == split)
    available = meta.loc[mask, "patch_id"].tolist()
    if not available:
        raise ValueError(f"No {split.upper()} patches found for country={country!r}")
    if n_samples > len(available):
        print(
            f"Warning: Requested {n_samples} samples but only {len(available)} available. Using all."
        )
        n_samples = len(available)

    # Sample patches randomly
    rng = random.Random(seed)
    sampled = rng.sample(available, k=n_samples)

    # Split into train and validation sets
    split_at = int(n_samples * frac)
    train_ids = set(sampled[:split_at])
    val_ids = set(sampled[split_at:])

    def _make_ds(keep_ids):
        """Helper function to create dataset with specific patch IDs."""
        return BENv2DataSet(
            data_dirs=datapath,
            img_size=img_size_ben,
            split=split,
            include_snowy=include_snowy,
            include_cloudy=include_cloudy,
            patch_prefilter=lambda pid: pid in keep_ids,
            transform=transform,
        )

    train_ds = _make_ds(train_ids)
    val_ds = _make_ds(val_ids)
    return train_ds, val_ds


def get_datasets(
    countries: List[str],
    permutation: List[int],
    split: str = "train",
    include_snowy: bool = False,
    include_cloudy: bool = False,
    samples_per_country: int = None,
    seed: int = 0,
    frac=0.8,
    model_module=None,  # Specify the model module
    use_saved_datasets: bool = False,  # New parameter
    saved_datasets_dir: str = '~/saved_datasets'  # New parameter
) -> Tuple[List[BENv2DataSet], List[BENv2DataSet]]:
    """
    Dynamically load datasets for the specified countries and permutation.
    
    This function loads datasets for multiple countries according to a permutation
    order, adjusting image size based on the model architecture.
    
    Args:
        countries: List of country names
        permutation: Order in which to load countries
        split: Data split ('train' or 'test')
        include_snowy: Whether to include snowy patches
        include_cloudy: Whether to include cloudy patches
        samples_per_country: Number of samples per country
        seed: Random seed for reproducibility
        frac: Fraction for train/val split
        model_module: Model architecture ('SpectralGPT' or 'SoftCon')
        use_saved_datasets: Whether to use pre-saved datasets
        saved_datasets_dir: Directory for saved datasets
    
    Returns:
        tuple: (train_datasets, val_datasets) lists for all countries
    """
    if use_saved_datasets:
        # Import the loader only when needed
        from saved_dataset_loader import load_country_datasets
        return load_country_datasets(countries, saved_datasets_dir)
    
    train_datasets: List[BENv2DataSet] = []
    val_datasets: List[BENv2DataSet] = []

    # Load datasets for each country in permutation order
    for idx in permutation:
        country = countries[idx]
        train_ds, val_ds = load_country_train_val(
            country,
            samples_per_country,
            seed,
            include_snowy=include_snowy,
            include_cloudy=include_cloudy,
            split=split,
            frac=frac,
            model_module=model_module,  # Pass model_module argument
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    return train_datasets, val_datasets


def default_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
    """
    Default evaluation metric: wraps sklearn.metrics.average_precision_score (micro average).
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
    
    Returns:
        dict: Dictionary containing micro average precision score
    """
    return {"micro_AP": average_precision_score(y_true, y_pred, average="micro")}


# ---------------------------------------------------------
# FINE-TUNING FROM SCRATCH (Full Replay)
# ---------------------------------------------------------
def test_finetuning_from_scratch(model: Any, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Continual Learning with full replay over permuted countries.
    
    This function implements a baseline continual learning strategy where:
    1. At each step, the model is reinitialized from base weights
    2. Training is performed on all seen countries (full replay)
    3. Evaluation is conducted on all seen countries
    4. LoRA parameters are saved for each step
    
    Key Features:
    - Full replay: Uses all data from previously seen countries
    - From scratch: Reinitializes model at each step
    - Comprehensive evaluation: Tests on all seen countries
    - Parameter saving: Saves LoRA and classifier weights
    
    Args:
        model: Base model to fine-tune
        params: Configuration dictionary containing experiment parameters
    
    Returns:
        pd.DataFrame: Results dataframe with metrics for each step
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    countries = params["countries"]
    permutation = params["permutation"]
    train_samples = params.get("train_samples")
    test_samples = params.get("test_samples")
    seed = params.get("seed", 0)
    batch_size = params.get("batch_size", 16)
    num_workers = params.get("num_workers", 4)
    epoch = params.get("epoch", 15)
    lr = float(params.get("lr", 1e-4))
    rank = params.get("r", 4)  # Default rank for LoRA
    model_module = params.get("model_module")  # Extract model_module from params

    # Load datasets for all countries
    train_sets, val_sets = get_datasets(
        countries,
        permutation,
        split="train",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=train_samples,
        seed=seed,
        frac=0.9,
        model_module=model_module,  # Pass model_module explicitly
    )
    test_sets, _ = get_datasets(
        countries,
        permutation,
        split="test",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=test_samples,
        seed=seed,
        frac=1,
        model_module=model_module,  # Pass model_module explicitly
    )

    # save unchanged lora_weights once

    # Compute an absolute directory under your home or your repo
    repo_dir = os.path.dirname(
        os.path.abspath(__file__)
    )  # e.g. .../anton/src/cl-lora/baseline
    save_dir = os.path.join(
        repo_dir, "saved_models"
    )  # e.g. .../anton/src/cl-lora/baseline/saved_models
    os.makedirs(save_dir, exist_ok=True)  # create it if neede
    path_to_file = "/home/anton/src/cl-lora/baseline/saved_models/lora_weights_baseline.safetensors"


    metrics_fn = params.get("metrics_fn", default_metrics)
    save_dir = params.get("save_dir")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    # Main loop: incrementally add countries
    for step in range(1, len(permutation) + 1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        log.info(f"Step {step}: Countries {seen_countries}")

        # Build training DataLoader with all seen countries
        concat_train = ConcatDataset(train_sets[:step])
        concat_val = ConcatDataset(val_sets[:step])
        train_loader = DataLoader(
            concat_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            concat_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Reinitialize the trainer for each step
        trainer = create_trainer(params)
        
        # Load fresh base model for each step
        if model_module == "SoftCon":
            base_model = load_model(r=rank)  # Load base model with pretrained weights
            pl_model = SoftConLightningModule(base_model, num_classes=19, lr=lr)
        elif model_module == "SpectralGPT":
            base_model = load_model(r=rank)
            pl_model = SpectralGPTLightningModule(base_model, num_classes=19, lr=lr)
        else:
            raise ValueError(f"Unknown model_module: {model_module}")

        # Train the model
        start_train = time.time()
        trainer.fit(pl_model, train_loader, val_loader)
        train_time = time.time() - start_train
        log.info(f"Training time: {train_time:.2f}s")

        # Save LoRA and FC parameters
        inner_model = getattr(pl_model, "model", pl_model)
        if hasattr(inner_model, "save_lora_parameters"):
            lora_path = os.path.join(save_dir, f"step{step}_lora.safetensors")
            fc_path = os.path.join(save_dir, f"step{step}_fc.safetensors")
            if trainer.is_global_zero:
                # Make sure to save LoRA and FC parameters only after parallelised training is finished
                inner_model.save_lora_parameters(lora_path)
                inner_model.save_fc_parameters(fc_path)
                log.info(f"LoRA saved to {lora_path}; FC saved to {fc_path}")
            else:
                inner_model.save_lora_parameters(lora_path)
                inner_model.save_fc_parameters(fc_path)
                log.info(
                    f"Skipping saving LoRA and FC parameters for step {step} (not global zero)"
                )

        # Save complete model
        if save_dir:
            path = os.path.join(save_dir, f"step{step}-{'-'.join(seen_countries)}.pkl")
            joblib.dump(pl_model, path)
            log.info(f"Model saved: {path}")

        # Evaluate on all seen countries
        eval_metrics: Dict[str, float] = {}
        start_eval = time.time()
        for pos, idx in enumerate(seen_idx):
            country = countries[idx]
            test_loader = DataLoader(
                test_sets[pos],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            m = eval_model(pl_model, test_loader)

            for name, val in m.items():
                eval_metrics[f"{country}_{name}"] = val
        eval_time = time.time() - start_eval
        log.info(f"Evaluation time: {eval_time:.2f}s")

        # Calculate mean metrics across all countries
        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items():
            _, met = k.split("_", 1)
            sums[met] += v
            counts[met] += 1
        mean_metrics = {f"mean_{m}": sums[m] / counts[m] for m in sums}

        # Compile results for this step
        row = {
            "step": step,
            "countries": tuple(seen_countries),
            "train_time_s": train_time,
            "eval_time_s": eval_time,
        }
        row.update(eval_metrics)
        row.update(mean_metrics)
        results.append(row)
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


# ---------------------------------------------------------
# CONTINUAL FINE-TUNING (No Replay)
# ---------------------------------------------------------
def test_continual_finetuning(model: Any, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Continual Learning without Replay: Train on new country, evaluate on all seen countries.
    
    This function implements a baseline continual learning strategy where:
    1. The model is trained incrementally on new countries
    2. No replay data is used (only current country data)
    3. Evaluation is conducted on all previously seen countries
    4. Model state is maintained across steps
    
    Key Features:
    - No replay: Only uses current country data for training
    - Incremental: Maintains model state across steps
    - Catastrophic forgetting: Expected to show forgetting of previous tasks
    - Comprehensive evaluation: Tests on all seen countries
    
    Args:
        model: Base model to fine-tune
        params: Configuration dictionary containing experiment parameters
    
    Returns:
        pd.DataFrame: Results dataframe with metrics for each step
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    countries = params["countries"]
    permutation = params["permutation"]
    train_samples = params.get("train_samples")
    test_samples = params.get("test_samples")
    seed = params.get("seed", 0)
    batch_size = params.get("batch_size", 16)
    num_workers = params.get("num_workers", 4)
    epoch = params.get("epoch", 15)
    lr = float(params.get("lr", 1e-4))
    rank = params.get("r", 4)  # Default rank for LoRA
    model_module = params.get("model_module")  # Extract model_module from params

    save_dir = params.get("save_dir")
    metrics_fn = params.get("metrics_fn", default_metrics)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load datasets for all countries
    train_sets, val_sets = get_datasets(
        countries,
        permutation,
        split="train",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=train_samples,
        seed=seed,
        model_module=model_module,  # Pass model_module explicitly
    )
    test_sets, _ = get_datasets(
        countries,
        permutation,
        split="test",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=test_samples,
        seed=seed,
        frac=1,
        model_module=model_module,  # Pass model_module explicitly
    )
    
    # Construct LightningModule
    if model_module == "SoftCon":
        pl_model = SoftConLightningModule(model, num_classes=19, lr=lr)
    elif model_module == "SpectralGPT":
        pl_model = SpectralGPTLightningModule(model, num_classes=19, lr=lr)
    else:
        raise ValueError(f"Unknown model_module: {model_module}")

    results = []
    prev_model_path = None
    
    # Main loop: incrementally add countries
    for step in range(1, len(permutation) + 1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        log.info(f"Step {step}: Countries {seen_countries}")

        # Train on new country DataLoader only
        ds_new = train_sets[step - 1]
        train_loader = DataLoader(
            ds_new, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        ds_new_val = val_sets[step - 1]
        val_loader = DataLoader(
            ds_new_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Reinitialize the trainer for each step
        trainer = create_trainer(params)

        # Reinitialize LightningModule for each step while retaining the trained model
        inner_model = pl_model.model
        if model_module == "SoftCon":
            pl_model = SoftConLightningModule(inner_model, num_classes=19, lr=lr)
        elif model_module == "SpectralGPT":
            pl_model = SpectralGPTLightningModule(inner_model, num_classes=19, lr=lr)
        else:
            raise ValueError(f"Unknown model_module: {model_module}")

        # Train the model
        start_train = time.time()
        trainer.fit(pl_model, train_loader, val_loader)
        train_time = time.time() - start_train
        log.info(f"Training time: {train_time:.2f}s")

        # Save LoRA and FC parameters
        inner_model = getattr(pl_model, "model", pl_model)
        if hasattr(inner_model, "save_lora_parameters"):
            lora_path = os.path.join(save_dir, f"step{step}_lora.safetensors")
            fc_path = os.path.join(save_dir, f"step{step}_fc.safetensors")
            if trainer.is_global_zero:
                # Make sure to save LoRA and FC parameters only after parallelised training is finished
                inner_model.save_lora_parameters(lora_path)
                inner_model.save_fc_parameters(fc_path)
                log.info(f"LoRA saved to {lora_path}; FC saved to {fc_path}")
            else:
                inner_model.save_lora_parameters(lora_path)
                inner_model.save_fc_parameters(fc_path)
                log.info(
                    f"Skipping saving LoRA and FC parameters for step {step} (not global zero)"
                )

        # Save model for this step
        if save_dir:
            path = os.path.join(save_dir, f"step{step}-{seen_countries[-1]}.pkl")
            joblib.dump(pl_model, path)
            prev_model_path = path
            log.info(f"Saved: {path}")

        # Evaluate on all seen countries
        eval_metrics = {}
        start_eval = time.time()
        for pos, idx in enumerate(seen_idx):
            country = countries[idx]
            test_loader = DataLoader(
                test_sets[pos],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            m = eval_model(pl_model, test_loader)
            for name, val in m.items():
                eval_metrics[f"{country}_{name}"] = val
        eval_time = time.time() - start_eval
        log.info(f"Eval time: {eval_time:.2f}s")

        # Calculate mean metrics across all countries
        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items():
            _, met = k.split("_", 1)
            sums[met] += v
            counts[met] += 1
        mean_metrics = {f"mean_{m}": sums[m] / counts[m] for m in sums}

        # Compile results for this step
        row = {
            "step": step,
            "countries": tuple(seen_countries),
            "train_time_s": train_time,
            "eval_time_s": eval_time,
        }
        row.update(eval_metrics)
        row.update(mean_metrics)
        results.append(row)
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


# ---------------------------------------------------------
# TASK TUNING for merging
# ---------------------------------------------------------
def test_task_tuning(model: Any, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Task-specific tuning: Train the base model independently on each country.
    
    This function trains independent LoRA adapters for each country, which can then
    be used as input for LoRA merging strategies. Each task is trained from scratch
    without any continual learning mechanisms.
    
    Key Features:
    - Independent training: Each country is trained separately
    - From scratch: Fresh model for each task
    - LoRA generation: Creates task-specific LoRA adapters
    - Individual evaluation: Tests each task independently
    
    Args:
        model: Base model to fine-tune
        params: Configuration dictionary containing experiment parameters
    
    Returns:
        pd.DataFrame: Results dataframe with metrics for each task
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    countries = params["countries"]
    permutation = params["permutation"]
    train_samples = params.get("train_samples")
    test_samples = params.get("test_samples")
    seed = params.get("seed", 0)
    batch_size = params.get("batch_size", 16)
    num_workers = params.get("num_workers", 4)
    epoch = params.get("epoch", 15)
    lr = float(params.get("lr", 1e-4))
    rank = params.get("r", 4)  # Default rank for LoRA
    model_module = params.get("model_module")  # Extract model_module from params

    save_dir = params.get("save_dir")
    metrics_fn = params.get("metrics_fn", default_metrics)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load datasets for all countries
    train_sets, val_sets = get_datasets(
        countries,
        permutation,
        split="train",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=train_samples,
        seed=seed,
        model_module=model_module,  # Pass model_module explicitly
    )
    test_sets, _ = get_datasets(
        countries,
        permutation,
        split="test",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=test_samples,
        seed=seed,
        frac=1,
        model_module=model_module,  # Pass model_module explicitly
    )

    results = []

    # Train each country independently
    for step in range(1, len(permutation) + 1):
        country_idx = permutation[step - 1]
        country = countries[country_idx]
        log.info(f"Task {step}: Training independently on {country}")

        # Load fresh base model for each task
        if model_module == "SoftCon":
            base_model = load_model(r=rank)  # Load base model with pretrained weights
            pl_model = SoftConLightningModule(base_model, num_classes=19, lr=lr)
        elif model_module == "SpectralGPT":
            base_model = load_model(r=rank)
            pl_model = SpectralGPTLightningModule(base_model, num_classes=19, lr=lr)
        else:
            raise ValueError(f"Unknown model_module: {model_module}")

        # Train on current country only
        train_loader = DataLoader(
            train_sets[step - 1],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_sets[step - 1],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Reinitialize the trainer for each task
        trainer = create_trainer(params)

        # Train the model
        start_train = time.time()
        trainer.fit(pl_model, train_loader, val_loader)
        train_time = time.time() - start_train
        log.info(f"Training time for {country}: {train_time:.2f}s")

        # Save LoRA and FC parameters
        inner_model = getattr(pl_model, "model", pl_model)
        if hasattr(inner_model, "save_lora_parameters") and save_dir:
            lora_path = os.path.join(save_dir, f"{country}_lora.safetensors")
            fc_path = os.path.join(save_dir, f"{country}_fc.safetensors")
            if trainer.is_global_zero:
                # Make sure to save LoRA and FC parameters only after parallelised training is finished
                inner_model.save_lora_parameters(lora_path)
                inner_model.save_fc_parameters(fc_path)
                log.info(f"LoRA saved to {lora_path}; FC saved to {fc_path}")
            else:
                inner_model.save_lora_parameters(lora_path)
                inner_model.save_fc_parameters(fc_path)
                log.info(
                    f"Skipping saving LoRA and FC parameters for task {step} (not global zero)"
                )

        # Save model for this specific task
        if save_dir:
            path = os.path.join(save_dir, f"{country}-independent.pkl")
            joblib.dump(pl_model, path)
            log.info(f"Model saved: {path}")

        # Evaluate only on the current task's test set
        eval_metrics = {}
        start_eval = time.time()
        test_loader = DataLoader(
            test_sets[step - 1],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        m = eval_model(pl_model, test_loader)
        for name, val in m.items():
            eval_metrics[f"{country}_{name}"] = val

        eval_time = time.time() - start_eval
        log.info(f"Evaluation time for {country}: {eval_time:.2f}s")

        # For task tuning, we only have metrics for the current country
        row = {
            "step": step,
            "country": country,
            "train_time_s": train_time,
            "eval_time_s": eval_time,
        }
        row.update(eval_metrics)
        results.append(row)
        log.info(f"Task {step} ({country}) Result: {row}")

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Configuration and main execution functions
# ---------------------------------------------------------
def main_from_config(config_path: str) -> pd.DataFrame:
    """
    Load configuration from file and execute the baseline experiment.
    
    This function handles dynamic module importing, configuration loading,
    and dispatches to the appropriate baseline strategy based on test_type.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
    
    Returns:
        pd.DataFrame: Results of the baseline experiment
    """
    with open(config_path, "r") as f:
        cfg = (
            yaml.safe_load(f)
            if config_path.endswith((".yml", ".yaml"))
            else json.load(f)
        )
    from importlib import import_module

    # Dynamically import the specified module
    model_module_name = cfg.get(
        "model_module", "SpectralGPT"
    )  # Default to 'SpectralGPT'
    logging.info(f"Using model module: {model_module_name}")
    model_module = import_module(model_module_name)

    # Import everything from the module into the global namespace
    globals().update(
        {
            name: getattr(model_module, name)
            for name in dir(model_module)
            if not name.startswith("_")
        }
    )

    params = cfg["params"]
    test_type = cfg["test_type"]

    # Dynamically update save_dir based on permutation
    permutation = params["permutation"]
    permutation_str = "_".join(map(str, permutation))
    params["save_dir"] = os.path.join(
        params["save_dir"], test_type, f"permutation_{permutation_str}"
    )
    os.makedirs(params["save_dir"], exist_ok=True)  # Ensure the directory exists

    # Configure logging to store a copy of the log in the save directory
    log_file = os.path.join(params["save_dir"], "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(log_file),  # Log to file
        ],
    )
    logging.info(f"Save directory set to: {params['save_dir']}")

    # Add model_module to params
    params["model_module"] = model_module_name

    # Dynamically set metrics function if specified
    if isinstance(params.get("metrics_fn"), str):
        mod, fn = params["metrics_fn"].rsplit(":", 1)
        params["metrics_fn"] = getattr(import_module(mod), fn)

    model = load_model(r=params.get("r", 4))  # Pass params to load_model
    
    # Dispatch to appropriate baseline strategy
    if test_type == "replay":
        return test_finetuning_from_scratch(model, params)
    elif test_type == "no_replay":
        return test_continual_finetuning(model, params)
    elif test_type == "task_tuning":
        return test_task_tuning(model, params)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


def setup_logging():
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Logger initialized")


def main():
    """
    Main entry point for the baseline script.
    
    Parses command line arguments and executes the baseline experiment
    based on the provided configuration file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Config path: {args.config}")

    # ← simple dump of every line in the config into the log
    with open(args.config, "r", encoding="utf-8") as f:
        logging.info("Config file contents:\n%s", f.read())

    print(main_from_config(args.config))


if __name__ == "__main__":
    main()
