import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import torch

import joblib
import pandas as pd
import pytorch_lightning as L
import yaml

# --- DataSet factory based on configilm / BENv2 ---
from configilm import util
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.metrics import average_precision_score
from torch.utils.data import ConcatDataset, DataLoader

# Pfade zu den Daten
datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}
util.MESSAGE_LEVEL = util.MessageLevel.INFO

# Disable xFormers for debuggin on CPU
if not torch.cuda.is_available():
    os.environ["XFORMERS_DISABLED"] = "1"


# -- Trainer Parameters ---
def create_trainer(params: Dict[str, Any]) -> L.Trainer:
    """
    Create and configure a central PyTorch Lightning Trainer with early stopping,
    checkpointing, and learning rate scheduling.
    """

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience= 5, # Stop after 5 epochs without improvement
        mode="min",  # Minimize the validation loss
        min_delta=0.01,  # Minimum change to qualify as an improvement
        check_on_train_epoch_end=True,  # Check after each training epoch
        verbose=True,
    )

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        # TODO specify right save directory
        # dirpath=params.get("save_dir", "./saved_models"),  # Directory to save checkpoints
        # filename="{country}",  # Placeholder for dynamic naming        save_top_k=1,  # Save only the best model
        mode="min",  # Minimize the validation loss
        verbose=True,
    )

    # Learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Define the trainer
    trainer = L.Trainer(
        max_epochs=params.get("epoch", 15),
        accelerator="auto",  # Automatically use GPU if available
        log_every_n_steps=50,
        strategy= L.strategies.DDPStrategy(find_unused_parameters=True),  # Allow unused parameters
        default_root_dir=params.get(
            "save_dir", "./saved_models"
        ),  # Directory for logs and checkpoints
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],  # Add callbacks
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
    Load country-specific train/val datasets. Dynamically adjusts transformations and img_size
    based on the model_module argument.
    """
    # Dynamically adjust transformations and img_size
    if model_module == "SoftCon":
        transform = train_transform if split == "train" else val_transform
        img_size = (14, 224, 224)  # SoftCon-specific img_size
    elif model_module == "SpectralGPT":
        transform = train_transform if split == "train" else val_transform
        img_size = (12, 128, 128)  # SpectralGPT-specific img_size
    else:
        raise ValueError(f"Unknown model_module: {model_module}")

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

    rng = random.Random(seed)
    sampled = rng.sample(available, k=n_samples)

    split_at = int(n_samples * frac)
    train_ids = set(sampled[:split_at])
    val_ids = set(sampled[split_at:])

    def _make_ds(keep_ids):
        return BENv2DataSet(
            data_dirs=datapath,
            img_size=img_size,
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
) -> Tuple[List[BENv2DataSet], List[BENv2DataSet]]:
    """
    Dynamically load datasets for the specified countries and permutation.
    Adjusts img_size based on the model_module.
    """
    train_datasets: List[BENv2DataSet] = []
    val_datasets: List[BENv2DataSet] = []

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
    """
    return {"micro_AP": average_precision_score(y_true, y_pred, average="micro")}


# --- Experiments ---
def test_finetuning_from_scratch(model: Any, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Continual Learning with full replay over permuted countries.
    Always starts training from the base model weights at each step.
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
    model_module = params.get("model_module")  # Extract model_module from params

    # Load datasets
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

    #save unchanged lora_weights once

    # (1) Compute an absolute directory under your home or your repo
    repo_dir = os.path.dirname(os.path.abspath(__file__))  # e.g. .../anton/src/cl-lora/baseline
    save_dir = os.path.join(repo_dir, "saved_models")  # e.g. .../anton/src/cl-lora/baseline/saved_models
    os.makedirs(save_dir, exist_ok=True)  # create it if needed

    # # (2) Build a fully‐qualified filename
    # out_file = os.path.join(save_dir, "lora_weights_baseline.safetensors")

    # # (3) Pass that to save_fc_parameters and save_lora_parameters
    # model.save_fc_parameters(out_file)
    # model.save_lora_parameters(out_file)
    path_to_file = "/home/anton/src/cl-lora/baseline/saved_models/lora_weights_baseline.safetensors"

    # # Build fully-qualified filenames
    # out_file = os.path.join(save_dir, "lora_weights_baseline.safetensors")  # LoRA weights file
    # path_to_classifier_file = os.path.join(save_dir, "classifier_weights_baseline.pth")  # Classifier weights file

    metrics_fn = params.get('metrics_fn', default_metrics)
    save_dir = params.get('save_dir')
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    results = []
    for step in range(1, len(permutation) + 1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        log.info(f"Step {step}: Countries {seen_countries}")

        # Build training DataLoader
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
        if model =="SoftCon":
            # Reload the base model weights
            base_model = load_model(r=4)  # Load base model with pretrained weights

            # Reinitialize the LightningModule for each step
            pl_model = SoftConLightningModule(
                base_model, embed_dim=768, num_classes=19, lr=lr
            )
            model = pl_model
        else:
            model.load_fc_parameters(path_to_file)
            model.load_lora_parameters(path_to_file)
        #model = load_model(r=4)
        # Train
        start_train = time.time()
        trainer.fit(pl_model, train_loader, val_loader)
        train_time = time.time() - start_train
        log.info(f"Training time: {train_time:.2f}s")

        # Save model
        if save_dir:
            path = os.path.join(save_dir, f"step{step}-{'-'.join(seen_countries)}.pkl")
            joblib.dump(pl_model, path)
            log.info(f"Model saved: {path}")

        # Eval on all seen
        eval_metrics: Dict[str, float] = {}
        start_eval = time.time()
        for idx in seen_idx:
            country = countries[idx]
            test_loader = DataLoader(
                test_sets[idx],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            m = eval_model(pl_model, test_loader)

            for name, val in m.items():
                eval_metrics[f"{country}_{name}"] = val
        eval_time = time.time() - start_eval
        log.info(f"Evaluation time: {eval_time:.2f}s")

        # Mean metrics
        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items():
            _, met = k.split("_", 1)
            sums[met] += v
            counts[met] += 1
        mean_metrics = {f"mean_{m}": sums[m] / counts[m] for m in sums}

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


def test_continual_finetuning(model: Any, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Continual Learning without Replay: Train on new country, evaluate on all seen countries.
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
    model_module = params.get("model_module")  # Extract model_module from params

    save_dir = params.get("save_dir")
    metrics_fn = params.get("metrics_fn", default_metrics)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load datasets
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
    prev_model_path = None
    for step in range(1, len(permutation) + 1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        log.info(f"Step {step}: Countries {seen_countries}")

        # Train on new country DataLoader
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

        # Extract the LoRA model (feature extractor) from the Sequential object
        lora_model = model[0]  # First part of the Sequential object

        # Reinitialize the LightningModule for each step
        pl_model = SoftConLightningModule(
            lora_model, embed_dim=768, num_classes=19, lr=lr
        )

        # Train
        start_train = time.time()
        trainer.fit(pl_model, train_loader, val_loader)
        train_time = time.time() - start_train
        log.info(f"Training time: {train_time:.2f}s")

        if save_dir:
            path = os.path.join(save_dir, f"step{step}-{seen_countries[-1]}.pkl")
            joblib.dump(pl_model, path)
            prev_model_path = path
            log.info(f"Saved: {path}")

        # Eval all seen
        eval_metrics = {}
        start_eval = time.time()
        for idx in seen_idx:
            country = countries[idx]
            test_loader = DataLoader(
                test_sets[idx],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            m = eval_model(pl_model, test_loader)
            for name, val in m.items():
                eval_metrics[f"{country}_{name}"] = val
        eval_time = time.time() - start_eval
        log.info(f"Eval time: {eval_time:.2f}s")

        # Calculate mean metrics
        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items():
            _, met = k.split("_", 1)
            sums[met] += v
            counts[met] += 1
        mean_metrics = {f"mean_{m}": sums[m] / counts[m] for m in sums}

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


def main_from_config(config_path: str) -> pd.DataFrame:
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

    # Add model_module to params
    params["model_module"] = model_module_name

    # Dynamically set metrics function if specified
    if isinstance(params.get("metrics_fn"), str):
        mod, fn = params["metrics_fn"].rsplit(":", 1)
        params["metrics_fn"] = getattr(import_module(mod), fn)

    test_type = cfg["test_type"]
    if test_type == "replay":
        model = load_model(4)  # Dynamically call load_model from the imported module
        return test_finetuning_from_scratch(model, params)
    elif test_type == "no_replay":
        model = load_model(4)  # Dynamically call load_model from the imported module
        return test_continual_finetuning(model, params)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Logger initialized")


def main():
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
