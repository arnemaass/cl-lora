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

# -------------------------------------------------------------------------
#  TODO Helper: functions that need to be implemented
# -------------------------------------------------------------------------

def load_lora_weights(task_idx: int):
    """Load LoRA weights for a specific task."""
    # Implementation depends on your LoRA weight storage format
    try:
        # Example implementation - adjust path and format as needed
        weights_path = f"checkpoints/task_{task_idx}/lora_weights.pt"
        return torch.load(weights_path)
    except FileNotFoundError:
        return None

def load_classifier_weights(task_idx: int):
    """Load classifier weights for a specific task."""
    # Implementation depends on your classifier weight storage format
    try:
        # Example implementation - adjust path and format as needed
        weights_path = f"checkpoints/task_{task_idx}/classifier_weights.pt"
        return torch.load(weights_path)
    except FileNotFoundError:
        return None


# -------------------------------------------------------------------------
#  MERGING FUNCTIONS
# -------------------------------------------------------------------------

def test_merging(test_type, params: Dict[str, Any]) -> pd.DataFrame:
    """
    TODO: Expected Behavior (to be tested):
    - load a stratified subset (5 to 10 %) for every orginal task
    - use the base model without wrapping LoRA_vit
    - create a LightningModule for the model
    - load the LoRA weights for all tasks
    - load the classifier weights for all tasks
    - for task i in countries:
        - start at task 2 
        - pool the tasks data and create train and validation sets
        - merge the classifiers based on sample size by task
        - TODO: Handled Externally: apply specific Merging Strategy:
            - merges the LoRA weights of tasks 
            - gradient based merging ( eg. LoRA Soups and ZipLora): train the classfier in the same process
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
    subset_fraction = params.get("subset_fraction", 0.1)  # 10% subset by default
    seed = params.get("seed", 0)
    batch_size = params.get("batch_size", 16)
    num_workers = params.get("num_workers", 4)
    epoch = params.get("epoch", 1)  # 1 epoch as specified
    lr = float(params.get("lr", 1e-4))  # lr 1e-4 as specified
    model_module = params.get("model_module")

    # Load full datasets first
    full_train_sets, full_val_sets = get_datasets(
        countries,
        permutation,
        split="train",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=None,  # Load all samples first
        seed=seed,
        frac=0.9,
        model_module=model_module,
    )
    
    test_sets, _ = get_datasets(
        countries,
        permutation,
        split="test",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=params.get("test_samples"),
        seed=seed,
        frac=1,
        model_module=model_module,
    )

    # Create stratified subsets (5-10%) for every original task
    train_subsets = []
    val_subsets = []
    for i, (train_set, val_set) in enumerate(zip(full_train_sets, full_val_sets)):
        train_subset_size = max(1, int(len(train_set) * subset_fraction))
        val_subset_size = max(1, int(len(val_set) * subset_fraction))
        
        train_subsets.append(sample_subset(train_set, train_subset_size, seed + i))
        val_subsets.append(sample_subset(val_set, val_subset_size, seed + i))
        
        log.info(f"Task {i+1} subset sizes: train={train_subset_size}/{len(train_set)}, "
                f"val={val_subset_size}/{len(val_set)}")

    save_dir = params.get('save_dir')
    if save_dir: 
        os.makedirs(save_dir, exist_ok=True)

    # Load base model without LoRA wrapping
    if model_module == "SpectralGPT":
        base_model = load_model(r=4, use_lora=False)  # TODO Base model without LoRA
    elif model_module == "SoftCon":
        base_model = load_model(r=4, use_lora=False)  # TODO Base model without LoRA
    else:
        raise ValueError(f"Unknown model_module: {model_module}")

    # Pre-load all LoRA weights and classifier weights for all tasks
    all_lora_weights = {}
    all_classifier_weights = {}
    
    for task_idx in range(1, len(countries) + 1):
        try:
            all_lora_weights[task_idx] = load_lora_weights(task_idx)
            all_classifier_weights[task_idx] = load_classifier_weights(task_idx)
            log.info(f"Loaded weights for task {task_idx}")
        except Exception as e:
            log.warning(f"Could not load weights for task {task_idx}: {e}")

    # Main loop: start at task 2 and increment
    results = []
    
    for current_task in range(2, len(permutation) + 1):
        seen_task_indices = permutation[:current_task]
        seen_countries = [countries[i] for i in seen_task_indices]
        log.info(f"Task {current_task}: Merging countries {seen_countries}")

        # Pool the task data and create train/validation sets
        pooled_train_sets = [train_subsets[i] for i in seen_task_indices]
        pooled_val_sets = [val_subsets[i] for i in seen_task_indices]
        
        # Calculate sample sizes for each task (for weighted merging)
        task_sample_sizes = [len(train_subsets[i]) for i in seen_task_indices]
        total_samples = sum(task_sample_sizes)
        task_weights = [size / total_samples for size in task_sample_sizes]
        
        log.info(f"Task weights based on sample sizes: {dict(zip(seen_countries, task_weights))}")

        # Create concatenated datasets
        concat_train = ConcatDataset(pooled_train_sets)
        concat_val = ConcatDataset(pooled_val_sets)
        
        train_loader = DataLoader(
            concat_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            concat_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Create LightningModule for the base model
        if model_module == "SpectralGPT":
            pl_model = SpectralGPTLightningModule(
                base_model, num_classes=19, lr=lr
            )
        elif model_module == "SoftCon":
            pl_model = SoftConLightningModule(
                base_model, num_classes=19, lr=lr
            )
        else:
            raise ValueError(f"Unknown model_module: {model_module}")

        # Apply specific merging strategy
        start_merge = time.time()
        
        # Collect LoRA weights and classifier weights for current tasks
        current_lora_weights = [all_lora_weights.get(i+1) for i in seen_task_indices]
        current_classifier_weights = [all_classifier_weights.get(i+1) for i in seen_task_indices]
        
        # TODO: 
        if test_type == "ZipLoRA":
            from ziplora import ZipLoRaMerge
            merged_model, merged_lora, merged_classifier = ZipLoRaMerge(
                pl_model, current_lora_weights, current_classifier_weights,
                task_weights, train_loader, val_loader, current_task, params
            )
        elif test_type == "LoRASoups":
            from LoRASoups import LoRASoupsMerge
            merged_model, merged_lora, merged_classifier = LoRASoupsMerge(
                pl_model, train_loader, val_loader, lora_head1, lora_head2, fc_head1, fc_head2, params
            )
        elif test_type == "LoRAHub":
            from LoRAHub import LoRAHubMerge
            # LoRAHub: first merge then train classifier
            merged_model, merged_lora, merged_classifier = LoRAHubMerge(
                pl_model, current_lora_weights, current_classifier_weights,
                task_weights, train_loader, val_loader, params
            )
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        merge_time = time.time() - start_merge
        log.info(f"Merge time for task {current_task}: {merge_time:.2f}s")

        # Save merged model
        if save_dir:
            path = os.path.join(save_dir, f"merged_task{current_task}-{'-'.join(seen_countries)}.pkl")
            joblib.dump({
                'model': merged_model,
                'lora_weights': merged_lora,
                'classifier_weights': merged_classifier
            }, path)
            log.info(f"Merged model saved: {path}")

        # Evaluate the merged model on the combined test set of all seen tasks
        eval_metrics: Dict[str, float] = {}
        start_eval = time.time()
        
        # Create combined test set for all seen tasks
        combined_test_sets = [test_sets[i] for i in seen_task_indices]
        combined_test = ConcatDataset(combined_test_sets)
        combined_test_loader = DataLoader(
            combined_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Evaluate on combined test set
        combined_metrics = eval_model(merged_model, combined_test_loader)
        for name, val in combined_metrics.items():
            eval_metrics[f"combined_{name}"] = val

        # Also evaluate on individual country test sets
        for idx in seen_task_indices:
            country = countries[idx]
            test_loader = DataLoader(
                test_sets[idx],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            country_metrics = eval_model(merged_model, test_loader)
            for name, val in country_metrics.items():
                eval_metrics[f"{country}_{name}"] = val

        eval_time = time.time() - start_eval
        log.info(f"Evaluation time for task {current_task}: {eval_time:.2f}s")

        # Calculate mean metrics across countries
        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items():
            if not k.startswith("combined_"):
                _, met = k.split("_", 1)
                sums[met] += v
                counts[met] += 1
        mean_metrics = {f"mean_{m}": sums[m] / counts[m] for m in sums}

        row = {
            "task": current_task,
            "countries": tuple(seen_countries),
            "merge_time_s": merge_time,
            "eval_time_s": eval_time,
            "num_tasks_merged": len(seen_task_indices),
        }
        row.update(eval_metrics)
        row.update(mean_metrics)
        results.append(row)
        
        log.info(f"Task {current_task} completed. Combined test accuracy: "
                f"{eval_metrics.get('combined_accuracy', 'N/A')}")

    return pd.DataFrame(results)

# ------------------------------------------------------------------
# - ANTONS VERSION
# ------------------------------------------------------------------
# --- Experiments --
# def test_merging(test_type, params: Dict[str, Any]) -> pd.DataFrame:
#     """
#     TODO: Expected Behavior (to be implemented):
#     - load a stratified subset (5 to 10 %) for every orginal task
#     - use the base model without wrapping LoRA_vit
#     - create a LightningModule for the model
#     - load the LoRA weights for all tasks
#     - load the classifier weights for all tasks
#     - for task i in countries:
#         - start at task 2 
#         - pool the tasks data and create train and validation sets
#         - merge the classifiers based on sample size by task
#         - TODO: Handled ExternallyApply: apply specific Merging Strategy:
#             - merges the LoRA weights of tasks 
#             - gradient based merging: train the classfier in the same process
#             - LoRAHub: first merge then train the classifier? 
#             - 1 epoch with lr 1e-4 
#             - return the merged model, lora_weights and classifier weights
#         - Evaluate the merged model on the combined test set of all tasks
#         - increment to the next task 
#     """
#     logging.basicConfig(level=logging.INFO)
#     log = logging.getLogger(__name__)

#     countries = params["countries"]
#     permutation = params["permutation"]
#     train_samples = params.get("train_samples")
#     test_samples = params.get("test_samples")
#     seed = params.get("seed", 0)
#     batch_size = params.get("batch_size", 16)
#     num_workers = params.get("num_workers", 4)
#     epoch = params.get("epoch", 15)
#     lr = float(params.get("lr", 1e-4))
#     model_module = params.get("model_module")  # Extract model_module from params

#     # Load datasets
#     train_sets, val_sets = get_datasets(
#         countries,
#         permutation,
#         split="train",
#         include_snowy=params.get("include_snowy", False),
#         include_cloudy=params.get("include_cloudy", False),
#         samples_per_country=train_samples,
#         seed=seed,
#         frac=0.9,
#         model_module=model_module,  # Pass model_module explicitly
#     )
#     test_sets, _ = get_datasets(
#         countries,
#         permutation,
#         split="test",
#         include_snowy=params.get("include_snowy", False),
#         include_cloudy=params.get("include_cloudy", False),
#         samples_per_country=test_samples,
#         seed=seed,
#         frac=1,
#         model_module=model_module,  # Pass model_module explicitly
#     )

#     save_dir = params.get('save_dir')
#     if save_dir: os.makedirs(save_dir, exist_ok=True)

#     # Load model
#     if model_module == "SoftCon":
#         model = load_model(r=4)  # load old model

#         pl_model = SpectralGPTLightningModule(
#             model, embed_dim=768, num_classes=19, lr=lr
#         )

#     elif model_module == "SpectralGPT":
#         model = load_model(r=4)  # load old model

#         pl_model = SpectralGPTLightningModule(
#             model, num_classes=19, lr=lr
#         )

#     else:
#         raise ValueError(f"Unknown model_module: {model_module}")

#     # Get first Model weights
#     # TODO load weigts for first task
#     lora_old = load_lora(1)

#     # -------------------------------------------------------------------------
#     #  Continual-learning loop with bounded replay memory
#     # -------------------------------------------------------------------------
#     results = []
#     memory_size = params.get("memory_size")  # total budget for old data
#     replay_train_sets, replay_val_sets = [], []  # what actually goes into replay

#     for step in range(2, len(permutation) + 1):
#         seen_idx = permutation[:step]
#         seen_countries = [countries[i] for i in seen_idx]
#         log.info(f"Step {step}: Countries {seen_countries}")

#         # ---------------------------------------------------------------------
#         #  Decide how many samples of *each* old task we may keep
#         # ---------------------------------------------------------------------
#         if step == 2:  # only one old task so far
#             share = memory_size
#         else:
#             share = math.ceil(memory_size / (step - 1))  # equal share per old task

#         # Build replay sets for ALL old tasks (step-1 of them)
#         replay_train_sets.clear()
#         replay_val_sets.clear()
#         for t in range(step - 1):
#             replay_train_sets.append(sample_subset(train_sets[t], share, seed))
#             replay_val_sets.append(sample_subset(val_sets[t], share, seed))

#         # New (current) task gets *all* its data
#         train_current = train_sets[step - 1]
#         val_current = val_sets[step - 1]

#         # ---------------------------------------------------------------------
#         #  Create loaders
#         # ---------------------------------------------------------------------
#         concat_train_old = ConcatDataset(replay_train_sets)
#         concat_val_old = ConcatDataset(replay_val_sets)
#         train_loader_old = DataLoader(
#             concat_train_old, batch_size=batch_size, shuffle=True, num_workers=num_workers
#         )
#         val_loader_old = DataLoader(
#             concat_val_old, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )

#         train_loader_new = DataLoader(
#             train_current, batch_size=batch_size, shuffle=True, num_workers=num_workers
#         )
#         val_loader_new = DataLoader(
#             val_current, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )

#         # TODO we need to load the new lora weights
#         lora_new = load_lora(step)
#         # TODO we need to load the 2 models correctly
#         # Reinitialize the trainer for each step
#         trainer = create_trainer(params)
#         if model_module == "SoftCon":
#             model_old = load_model(r=4)  # load old model
#             model_new = load_model(r=4)  # load new model

#             pl_model_old = SoftConLightningModule(
#                 base_model, num_classes=19, lr=lr
#             )
#             pl_model_new = SoftConLightningModule(
#                 base_model, num_classes=19, lr=lr
#             )

#         elif model_module == "SpectralGPT":
#             model_old= load_model(r=4) #load old model
#             model_new = load_model(r=4) # load new model

#             pl_model_old = SpectralGPTLightningModule(
#                 model_old, num_classes=19, lr=lr
#             )
#             pl_model_new = SpectralGPTLightningModule(
#                 model_new, num_classes=19, lr=lr
#             )
#         else:
#             raise ValueError(f"Unknown model_module: {model_module}")

#         # TODO here we call the merging methods
#         # Merge
#         start_merge = time.time()
#         if test_type=="ZipLoRA":
#             from ziplora import ZipLoRaMerge
#             pl_model, lora_old = ZipLoRaMerge(pl_model,lora_old,lora_new,train_loader_old,train_loader_new,val_loader_old,val_loader_new,step,params)
#         if test_type == "LoRASoups":
#             pl_model = LoRASoupsMerge()
#         if test_type == "LoRAHub":
#             pl_model = LoRAHubMerge()
#         else:
#             raise ValueError(f"Unknown test_type: {test_type}")

#         merge_time = time.time() - start_merge
#         log.info(f"Merge time: {merge_time:.2f}s")

#         # Save model
#         if save_dir:
#             path = os.path.join(save_dir, f"step{step}-{'-'.join(seen_countries)}.pkl")
#             joblib.dump(pl_model, path)
#             log.info(f"Model saved: {path}")

#         # Eval on all seen
#         eval_metrics: Dict[str, float] = {}
#         start_eval = time.time()
#         for idx in seen_idx:
#             country = countries[idx]
#             test_loader = DataLoader(
#                 test_sets[idx],
#                 batch_size=batch_size,
#                 shuffle=False,
#                 num_workers=num_workers,
#             )
#             m = eval_model(pl_model, test_loader)

#             for name, val in m.items():
#                 eval_metrics[f"{country}_{name}"] = val
#         eval_time = time.time() - start_eval
#         log.info(f"Evaluation time: {eval_time:.2f}s")

#         # Mean metrics
#         sums, counts = defaultdict(float), defaultdict(int)
#         for k, v in eval_metrics.items():
#             _, met = k.split("_", 1)
#             sums[met] += v
#             counts[met] += 1
#         mean_metrics = {f"mean_{m}": sums[m] / counts[m] for m in sums}

#         row = {
#             "step": step,
#             "countries": tuple(seen_countries),
#             "train_time_s": merge_time,
#             "eval_time_s": eval_time,
#         }
#         row.update(eval_metrics)
#         row.update(mean_metrics)
#         results.append(row)
#         log.info(f"Result: {row}")

#     return pd.DataFrame(results)

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
