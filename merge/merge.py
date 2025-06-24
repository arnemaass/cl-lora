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
from torch.utils.data import ConcatDataset, DataLoader, Subset
import math

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
    transform = train_transform if split == "train" else val_transform
    if model_module == "SoftCon":
        img_size_ben = (14, 224, 224)  # SoftCon-specific img_size for BENv2 loader
    elif model_module == "SpectralGPT":
        img_size_ben = (14, 128, 128)  # SpectralGPT-specific img_size for BENv2 loader
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

    # -------------------------------------------------------------------------
    #  Helper: randomly pick 'n' indices from a dataset
    # -------------------------------------------------------------------------


def sample_subset(dataset, n, seed):
    """Return a Subset of size min(n, len(dataset))."""
    if n >= len(dataset):
        return dataset  # keep the whole set
    g = torch.Generator().manual_seed(seed)  # reproducible
    idx = torch.randperm(len(dataset), generator=g)[: n]

    return Subset(dataset, idx.tolist())


# --- Experiments ---
def test_merging(test_type, params: Dict[str, Any]) -> pd.DataFrame:
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

    save_dir = params.get('save_dir')
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    #  Continual-learning loop with bounded replay memory
    # -------------------------------------------------------------------------
    results = []
    memory_size = params.get("memory_size")  # total budget for old data
    replay_train_sets, replay_val_sets = [], []  # what actually goes into replay

    for step in range(2, len(permutation) + 1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        log.info(f"Step {step}: Countries {seen_countries}")

        # ---------------------------------------------------------------------
        #  Decide how many samples of *each* old task we may keep
        # ---------------------------------------------------------------------
        if step == 2:  # only one old task so far
            share = memory_size
        else:
            share = math.ceil(memory_size / (step - 1))  # equal share per old task

        # Build replay sets for ALL old tasks (step-1 of them)
        replay_train_sets.clear()
        replay_val_sets.clear()
        for t in range(step - 1):
            replay_train_sets.append(sample_subset(train_sets[t], share, seed))
            replay_val_sets.append(sample_subset(val_sets[t], share, seed))

        # New (current) task gets *all* its data
        train_current = train_sets[step - 1]
        val_current = val_sets[step - 1]

        # ---------------------------------------------------------------------
        #  Create loaders
        # ---------------------------------------------------------------------
        concat_train_old = ConcatDataset(replay_train_sets)
        concat_val_old = ConcatDataset(replay_val_sets)
        train_loader_old = DataLoader(
            concat_train_old, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader_old = DataLoader(
            concat_val_old, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        train_loader_new = DataLoader(
            train_current, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader_new = DataLoader(
            val_current, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # TODO we need to load the 2 models correctly
        # Reinitialize the trainer for each step
        trainer = create_trainer(params)
        if model_module == "SoftCon":
            model_old = load_model(r=4)  # load old model
            model_new = load_model(r=4)  # load new model

            pl_model_old = SoftConLightningModule(
                base_model, num_classes=19, lr=lr
            )
            pl_model_new = SoftConLightningModule(
                base_model, num_classes=19, lr=lr
            )

        elif model_module == "SpectralGPT":
            model_old= load_model(r=4) #load old model
            model_new = load_model(r=4) # load new model

            pl_model_old = SpectralGPTLightningModule(
                model_old, num_classes=19, lr=lr
            )
            pl_model_new = SpectralGPTLightningModule(
                model_new, num_classes=19, lr=lr
            )
        else:
            raise ValueError(f"Unknown model_module: {model_module}")

        # TODO here we call the merging methods
        # Merge
        start_merge = time.time()
        if test_type=="ZipLoRA":
            from ziplora import ZipLoRaMerge
            pl_model = ZipLoRaMerge(pl_model_old,pl_model_new,train_loader_old,train_loader_new,val_loader_old,val_loader_new,step,params)
        if test_type == "LoRASoups":
            pl_model = LoRASoupsMerge()
        if test_type == "LoRAHub":
            pl_model = LoRAHubMerge()
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        merge_time = time.time() - start_merge
        log.info(f"Merge time: {merge_time:.2f}s")

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
            "train_time_s": merge_time,
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
    test_merging(test_type, params)

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
