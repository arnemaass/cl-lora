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
import sys

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

# Dynamically add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../baseline")
from baseline import (
    datapath,
    create_trainer,
    get_datasets,
    load_country_train_val,
    load_country_train_val,
    default_metrics,
)
from SpectralGPT import (
    load_model,
    eval_model,
)

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
    TODO: Expected Behavior (to be implemented):
    - load a stratified subset (5 to 10 %) for every orginal task
    - use the base model without wrapping LoRA_vit
    - create a LightningModule for the model
    - load the LoRA weights for all tasks
    - load the classifier weights for all tasks
    - for task i in countries:
        - start at task 2 
        - pool the tasks data and create train and validation sets
        - merge the classifiers based on sample size by task
        - TODO: Apply Specific Merging Strategy:
            - merges the LoRA weights of tasks 
            - gradient based merging: train the classfier in the same process
            - LoRAHub: first merge then train the classifier? 
            - 1 epoch with lr 1e-4 
            - return the merged model, lora_weights and classifier weights
        - Evaluate the merged model on the combined test set of all tasks
        - increment to the next task 
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

    # Load model
    if model_module == "SoftCon":
        model = load_model(r=4)  # load old model

        pl_model = SpectralGPTLightningModule(
            model, embed_dim=768, num_classes=19, lr=lr
        )

    elif model_module == "SpectralGPT":
        model = load_model(r=4)  # load old model

        pl_model = SpectralGPTLightningModule(
            model, num_classes=19, lr=lr
        )

    else:
        raise ValueError(f"Unknown model_module: {model_module}")

    # Get first Model weights
    # TODO load weigts for first task
    lora_old = load_lora(1)

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

        # TODO we need to load the new lora weights
        lora_new = load_lora(step)
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
            pl_model, lora_old = ZipLoRaMerge(pl_model,lora_old,lora_new,train_loader_old,train_loader_new,val_loader_old,val_loader_new,step,params)
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

# ------------------------------------------------------------------
# 
# ------------------------------------------------------------------

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
