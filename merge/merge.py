"""
merge.py
Main script for LoRA adapter merging experiments in continual learning scenarios.

This script implements and evaluates different LoRA merging strategies:
1. Continual merging: Incrementally merge new task adapters with previous results
2. From-scratch merging: Merge all original task adapters in a single step

Supported merging methods:
- LoRAHub: Gradient-free optimization for optimal LoRA weight combinations
- LoRASoups: Parametrization-based merging with learnable coefficients
- ZipLoRA: Alternative merging approach with cosine similarity

The script handles:
- Loading pre-trained LoRA adapters and classifier heads
- Dataset management and stratified sampling
- Model evaluation on individual and combined test sets
- Results logging and model saving

Dependencies
------------
• torch, pandas, joblib, yaml, sklearn
• Custom SpectralGPT and baseline modules
• LoRA merging implementations (LoRAHub, LoRASoups, ZipLoRA)
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict

import joblib
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

# --- DataSet factory based on configilm / BENv2 ---
from torch.utils.data import ConcatDataset, DataLoader, Subset

# Dynamically add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../baseline")
# # Import transforms along with other SpectralGPT components
from SpectralGPT import (
    # train_transform,
    # val_transform,
    SpectralGPTLightningModule,
    eval_model,
    load_model,
)

from baseline import (
    get_datasets,
)

# -------------------------------------------------------------------------
#  Helper: randomly pick 'n' indices from a dataset
# -------------------------------------------------------------------------


def sample_stratified_subset(dataset, n, seed):
    """
    Return a stratified Subset of size min(n, len(dataset)) maintaining class proportions.
    
    Args:
        dataset: Dataset to sample from
        n: Number of samples to select
        seed: Random seed for reproducibility
    
    Returns:
        Subset: Stratified subset of the original dataset
    """
    if n >= len(dataset):
        return dataset

    # Get all labels
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)

    indices = list(range(len(dataset)))
    try:
        selected_indices, _ = train_test_split(
            indices, train_size=n, stratify=labels, random_state=seed
        )
    except ValueError:  # Fall back to random if stratification fails
        g = torch.Generator().manual_seed(seed)
        selected_indices = torch.randperm(len(dataset), generator=g)[:n].tolist()
        logging.warning("Stratified sampling failed, falling back to random sampling.")

    return Subset(dataset, selected_indices)


# -------------------------------------------------------------------------
#  Helpers: load the weights
# -------------------------------------------------------------------------


def load_lora_weights(country: str, base_path: str = None):
    """
    Load LoRA weights for a specific country.
    
    Args:
        country: Country name for which to load LoRA weights
        base_path: Base path for weight files (optional)
    
    Returns:
        dict: LoRA state dictionary or None if not found
    """
    try:
        from safetensors.torch import load_file

        if base_path is None:
            base_path = "/faststorage/continual_low_rank_adaptation_of_remote_sensing_foundation_models/SpectralGPT/saved_models/epoch15/task_tuning"

        lora_path = os.path.join(base_path, f"{country}_lora.safetensors")

        if os.path.exists(lora_path):
            logging.info(f"Loading LoRA weights from: {lora_path}")
            return load_file(lora_path)
        else:
            logging.warning(f"LoRA weights not found: {lora_path}")
            return None

    except (FileNotFoundError, ImportError) as e:
        logging.warning(f"Could not load LoRA weights for {country}: {e}")
        return None


def load_classifier_weights(country: str, base_path: str = None):
    """
    Load classifier weights for a specific country.
    
    Args:
        country: Country name for which to load classifier weights
        base_path: Base path for weight files (optional)
    
    Returns:
        dict: Classifier state dictionary or None if not found
    """
    try:
        from safetensors.torch import load_file

        if base_path is None:
            base_path = "/faststorage/continual_low_rank_adaptation_of_remote_sensing_foundation_models/SpectralGPT/saved_models/epoch15/task_tuning"

        fc_path = os.path.join(base_path, f"{country}_fc.safetensors")

        if os.path.exists(fc_path):
            logging.info(f"Loading classifier weights from: {fc_path}")
            return load_file(fc_path)
        else:
            logging.warning(f"Classifier weights not found: {fc_path}")
            print(f"Classifier weights not found: {fc_path}")
            return None

    except (FileNotFoundError, ImportError) as e:
        logging.warning(f"Could not load classifier weights for {country}: {e}")
        return None


def load_weights_for_permutation(countries, permutation, base_path: str = None):
    """
    Load weights for countries according to permutation order.
    
    Args:
        countries: List of country names
        permutation: Order in which to load countries
        base_path: Base path for weight files (optional)
    
    Returns:
        tuple: (all_lora_weights, all_classifier_weights) lists
    """
    all_lora_weights = []
    all_classifier_weights = []
    print(permutation)
    print(countries)

    # Load weights according to permutation order
    for country_idx in permutation:
        if country_idx < len(countries):
            country = countries[country_idx]

            logging.info(
                f"Loading weights for country: {country} (permutation index: {country_idx})"
            )
            print(
                f"Loading weights for country: {country} (permutation index: {country_idx})"
            )
            lora_weights = load_lora_weights(country, base_path)
            classifier_weights = load_classifier_weights(country, base_path)

            if lora_weights is not None:
                all_lora_weights.append(lora_weights)
            else:
                logging.warning(f"No LoRA weights found for {country}")
                print("no weights")

            if classifier_weights is not None:
                all_classifier_weights.append(classifier_weights)
            else:
                logging.warning(f"No classifier weights found for {country}")

    return all_lora_weights, all_classifier_weights


def sample_subset(dataset, n, seed):
    """
    Return a Subset of size min(n, len(dataset)).
    
    Args:
        dataset: Dataset to sample from
        n: Number of samples to select
        seed: Random seed for reproducibility
    
    Returns:
        Subset: Random subset of the original dataset
    """
    print(dataset)
    if n >= len(dataset):
        return dataset  # keep the whole set
    g = torch.Generator().manual_seed(seed)  # reproducible
    idx = torch.randperm(len(dataset), generator=g)[:n]

    return Subset(dataset, idx.tolist())


# -------------------------------------------------------------------------
#  MERGING FUNCTIONS
# -------------------------------------------------------------------------


def test_continual_merging(test_type, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Continual merging: sequentially merge models one task at a time.
    
    This function implements incremental continual learning where:
    1. Start with the first task's adapter
    2. For each new task, merge the previous merged adapter with the new task's adapter
    3. Update the previous merged adapter to be the result of this merge
    4. Evaluate on all seen tasks after each merge
    
    Key Features:
    - Memory replay: Maintains a subset of old task data for replay
    - Incremental merging: Each merge combines previous result + new task
    - Comprehensive evaluation: Tests on individual and combined task performance
    
    Args:
        test_type: Type of merging method ('LoRAHub', 'LoRASoups', 'ZipLoRA')
        params: Configuration dictionary containing all experiment parameters
    
    Returns:
        pd.DataFrame: Results dataframe with metrics for each task
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    countries = params["countries"]
    permutation = params["permutation"]
    subset_fraction = params.get("subset_fraction", 0.1)
    seed = params.get("seed", 0)
    batch_size = params.get("batch_size", 16)
    num_workers = params.get("num_workers", 4)
    epoch = params.get("epoch", 1)
    lr = float(params.get("lr", 1e-4))
    model_module = params.get("model_module")

    # Load datasets for all countries
    full_train_sets, full_val_sets = get_datasets(
        countries,
        permutation,
        split="train",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=params.get("train_samples"),
        seed=seed,
        frac=0.9,
        model_module=model_module,
        use_saved_datasets=params.get("use_saved_datasets", False),
        saved_datasets_dir=params.get("saved_datasets_dir", "~/saved_datasets"),
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
        use_saved_datasets=params.get("use_saved_datasets", False),
        saved_datasets_dir=params.get("saved_datasets_dir", "~/saved_datasets"),
    )

    # Create stratified subsets for training
    train_subsets = []
    val_subsets = []
    for i, (train_set, val_set) in enumerate(zip(full_train_sets, full_val_sets)):
        train_subset_size = max(1, int(len(train_set)))
        val_subset_size = max(1, int(len(val_set)))

        train_subsets.append(
            sample_stratified_subset(train_set, train_subset_size, seed + i)
        )
        val_subsets.append(sample_stratified_subset(val_set, val_subset_size, seed + i))

    save_dir = params.get("save_dir")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load base model without LoRA wrapping
    if model_module == "SpectralGPT":
        base_model = load_model(r=4, use_lora=False)
    elif model_module == "SoftCon":
        base_model = load_model(r=4, use_lora=False)
    else:
        raise ValueError(f"Unknown model_module: {model_module}")

    # Pre-load all weights according to permutation order
    all_lora_weights, all_classifier_weights = load_weights_for_permutation(
        countries, permutation, params.get("weight_base_path")
    )

    # Initialize with first task
    previous_lora_weights = [all_lora_weights[0]]
    previous_classifier_weights = [all_classifier_weights[0]]

    results = []
    memory_size = params.get("memory_size")  # total budget for old data
    replay_train_sets, replay_val_sets = [], []  # what actually goes into replay

    # Continual merging loop: start at task 2
    for current_task in range(2, len(permutation) + 1):
        seen_task_indices = permutation[:current_task]
        seen_countries = [countries[i] for i in seen_task_indices]
        log.info(
            f"Task {current_task}: Adding {countries[permutation[current_task - 1]]} to previous merged model"
        )

        # Calculate memory allocation for replay
        if current_task == 2:  # only one old task so far
            share = memory_size
        else:
            share = math.ceil(
                memory_size / (current_task - 1)
            )  # equal share per old task

        # Prepare replay data from previous tasks
        replay_train_sets.clear()
        replay_val_sets.clear()
        for t in range(current_task - 1):
            replay_train_sets.append(sample_subset(full_train_sets[t], share, seed))
            replay_val_sets.append(sample_subset(full_val_sets[t], share, seed))

        train_current = full_train_sets[current_task - 1]
        val_current = full_val_sets[current_task - 1]

        # Get new task weights
        new_lora_weights = all_lora_weights[current_task - 1]
        new_classifier_weights = all_classifier_weights[current_task - 1]

        # Create data loaders for training
        new_train_loader = DataLoader(
            train_current, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        new_val_loader = DataLoader(
            val_current, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        old_train_loader = DataLoader(
            ConcatDataset(replay_train_sets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Create LightningModule for training
        if model_module == "SpectralGPT":
            pl_model = SpectralGPTLightningModule(base_model, num_classes=19, lr=lr)
        elif model_module == "SoftCon":
            pl_model = SpectralGPTLightningModule(
                base_model, embed_dim=768, num_classes=19, lr=lr
            )
        else:
            raise ValueError(f"Unknown model_module: {model_module}")

        # Perform merging based on test_type
        start_merge = time.time()

        if test_type == "LoRASoups":
            from LoRASoups import LoraSoupsMerge_continual

            # Merge previous merged weights with new task weights
            lora_heads_to_merge = previous_lora_weights + [new_lora_weights]
            classifier_heads_to_merge = previous_classifier_weights + [
                new_classifier_weights
            ]

            merged_model, merged_lora, merged_classifier = LoraSoupsMerge_continual(
                pl_model=pl_model,
                train_loader_new=new_train_loader,  # Train on new task data
                train_loader_old=old_train_loader,
                lora_heads=lora_heads_to_merge,
                classifier_heads=classifier_heads_to_merge,
                mode="learnable",
                num_epochs=epoch,
                lr=lr,
                current_task=current_task,
            )
            # Update previous to be the merged result (this is the key difference!)
            previous_lora_weights = [merged_lora]
            previous_classifier_weights = [merged_classifier]

        elif test_type == "ZipLoRA":
            from ziplora import ZipLoRaMerge

            # Merge previous merged weights with new task weights
            lora_heads_to_merge = previous_lora_weights + [new_lora_weights]
            classifier_heads_to_merge = previous_classifier_weights + [
                new_classifier_weights
            ]

            merged_model, merged_lora, merged_classifier = ZipLoRaMerge(
                pl_model=pl_model,
                train_loader_new=new_train_loader,  # Train on new task data
                train_loader_old=old_train_loader,
                lora_heads=lora_heads_to_merge,
                classifier_heads=classifier_heads_to_merge,
                step=current_task,
                num_epochs=epoch,
                lr=lr,
                d_cos=1,
            )

            # Update previous to be the merged result (this is the key difference!)
            previous_lora_weights = [merged_lora]
            previous_classifier_weights = [merged_classifier]

        elif test_type == "LoRAHub":
            # Similar pattern for LoRAHub
            from lorahub.lorahub_learning import LoraHubMerge_continual
            lora_heads_to_merge = previous_lora_weights + [new_lora_weights]
            classifier_heads_to_merge = previous_classifier_weights + [
                new_classifier_weights
            ]
            
            
            merged_model, merged_lora, merged_classifier = LoraHubMerge_continual(
                pl_model=pl_model,
                train_loader_new=new_train_loader,  # Train on new task data
                train_loader_old=old_train_loader,
                lora_heads=lora_heads_to_merge,
                classifier_heads=classifier_heads_to_merge,
                num_epochs=epoch,
                lr=lr,
                current_task=current_task,
            )
            
            previous_lora_weights = [merged_lora]
            previous_classifier_weights = [merged_classifier]
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        merge_time = time.time() - start_merge
        log.info(f"Merge time for task {current_task}: {merge_time:.2f}s")

        # Save merged model
        if save_dir:
            path = os.path.join(
                save_dir, f"continual_task{current_task}-{'-'.join(seen_countries)}.pkl"
            )
            joblib.dump(
                {
                    "lora_weights": merged_lora,
                    "classifier_weights": merged_classifier,
                    "task": current_task,
                    "countries": seen_countries,
                    "merge_time_s": merge_time,
                },
                path,
            )
            log.info(f"Continual merged model saved: {path}")

        # Evaluation on all seen tasks
        eval_metrics: Dict[str, float] = {}
        start_eval = time.time()

        # Create combined test set for all seen tasks
        # The test_sets list is already permuted. We just need the first `current_task` sets.
        combined_test_sets = test_sets[:current_task]
        combined_test = ConcatDataset(combined_test_sets)
        combined_test_loader = DataLoader(
            combined_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Evaluate on combined test set
        combined_metrics = eval_model(merged_model, combined_test_loader)
        for name, val in combined_metrics.items():
            eval_metrics[f"combined_{name}"] = val

        # Also evaluate on individual country test sets
        # Iterate through the position (0 to current_task-1) and the permuted index simultaneously.
        for pos, idx in enumerate(seen_task_indices):
            country = countries[idx]  # Get country name using the permuted index
            test_loader = DataLoader(
                test_sets[pos],  # Get dataset using the position
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            country_metrics = eval_model(merged_model, test_loader)
            for name, val in country_metrics.items():
                eval_metrics[f"{country}_{name}"] = val

        eval_time = time.time() - start_eval
        log.info(f"Evaluation time for task {current_task}: {eval_time:.2f}s")

        # Calculate mean metrics across all individual countries
        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items():
            if not k.startswith("combined_"):
                _, met = k.split("_", 1)
                sums[met] += v
                counts[met] += 1
        mean_metrics = {f"mean_{m}": sums[m] / counts[m] for m in sums}

        # Compile results for this task
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

        log.info(
            f"Task {current_task} completed. Combined test accuracy: "
            f"{eval_metrics.get('combined_accuracy', 'N/A')}"
        )
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


# -------------------------------------------------------------------------
# Can only be run for LoraHub and LoraSoups since it merges more then two tasks
# -------------------------------------------------------------------------
def test_merging_from_scratch(test_type, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Merges all tasks seen so far from scratch in each step of the loop.
    
    This function implements from-scratch merging where:
    1. At each step T, take the original adapters for tasks 1..T
    2. Merge all original adapters together in a single operation
    3. Evaluate on all seen tasks after each merge
    
    Key Features:
    - From-scratch approach: Always merges original adapters, not previous results
    - Complete data usage: Uses all available data from all seen tasks
    - Comprehensive evaluation: Tests on individual and combined task performance
    
    Args:
        test_type: Type of merging method ('LoRAHub', 'LoRASoups', 'ZipLoRA')
        params: Configuration dictionary containing all experiment parameters
    
    Returns:
        pd.DataFrame: Results dataframe with metrics for each task
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    countries = params["countries"]
    permutation = params["permutation"]
    subset_fraction = params.get("subset_fraction", 0.1)
    seed = params.get("seed", 0)
    batch_size = params.get("batch_size", 16)
    num_workers = params.get("num_workers", 4)
    epoch = params.get("epoch", 1)
    lr = float(params.get("lr", 1e-4))
    model_module = params.get("model_module")

    # Load full datasets for all countries
    full_train_sets, _ = get_datasets(
        countries,
        permutation,
        split="train",
        include_snowy=params.get("include_snowy", False),
        include_cloudy=params.get("include_cloudy", False),
        samples_per_country=params.get("train_samples"),
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

    # Create training subsets for every original task
    train_subsets = []
    for i, train_set in enumerate(full_train_sets):
        # subset_size = max(1, int(len(train_set) * subset_fraction))
        # train_subsets.append(sample_stratified_subset(train_set, subset_size, seed + i))
        # log.info(f"Task {i + 1} subset size: {subset_size}/{len(train_set)}")
        train_subsets.append(train_set)
        log.info(f"Task {i + 1} using full train set with size: {len(train_set)}")

    save_dir = params.get("save_dir")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load base model without LoRA wrapping
    if model_module == "SpectralGPT":
        base_model = load_model(r=4, use_lora=False)
    elif model_module == "SoftCon":
        base_model = load_model(r=4, use_lora=False)
    else:
        raise ValueError(f"Unknown model_module: {model_module}")

    # Pre-load all weights according to permutation order
    all_lora_weights, all_classifier_weights = load_weights_for_permutation(
        countries, permutation, params.get("weight_base_path")
    )

    log.info(f"Successfully loaded weights for {len(all_lora_weights)} countries")

    # Main loop: start at task 2 and increment
    results = []

    for current_task_num in range(2, len(permutation) + 1):
        seen_task_indices = permutation[:current_task_num]
        seen_countries = [countries[i] for i in seen_task_indices]
        log.info(
            f"Task {current_task_num}: Merging countries {seen_countries} from scratch"
        )

        # Pool all data from all tasks seen so far
        all_seen_data = [train_subsets[i] for i in seen_task_indices]

        # Create combined data loader for all seen tasks
        combined_train_loader = DataLoader(
            ConcatDataset(all_seen_data),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Create a fresh LightningModule for each from-scratch merge
        if model_module == "SpectralGPT":
            pl_model = SpectralGPTLightningModule(base_model, num_classes=19, lr=lr)
        elif model_module == "SoftCon":
            pl_model = SpectralGPTLightningModule(
                base_model, embed_dim=768, num_classes=19, lr=lr
            )
        else:
            raise ValueError(f"Unknown model_module: {model_module}")

        # Apply specific merging strategy
        start_merge = time.time()

        # Get the original weights for all current tasks
        lora_heads_to_merge = [all_lora_weights[i] for i in seen_task_indices]
        classifier_heads_to_merge = [
            all_classifier_weights[i] for i in seen_task_indices
        ]

        if test_type == "LoRASoups":
            from LoRASoups import LoraSoupsMerge_from_scratch

            merged_model, merged_lora, merged_classifier = LoraSoupsMerge_from_scratch(
                pl_model=pl_model,
                train_loader=combined_train_loader,  # Pass all data in one loader
                lora_heads=lora_heads_to_merge,
                classifier_heads=classifier_heads_to_merge,
                mode="learnable",
                num_epochs=epoch,
                lr=lr,
                current_task=current_task_num,
            )
        elif test_type == "ZipLoRA":
            from ziplora import ZipLoRaMerge

            merged_model, merged_lora, merged_classifier = ZipLoRaMerge(
                pl_model=pl_model,
                train_loader_old=combined_train_loader,  # All data
                train_loader_new=empty_loader,  # Empty loader here
                lora_heads=lora_heads_to_merge,
                classifier_heads=classifier_heads_to_merge,
                step=current_task_num,
                num_epochs=epoch,
                lr=lr,
                d_cos=1,
            )
        elif test_type == "LoRAHub":
            # Add LoRAHub implementation
            pass
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        merge_time = time.time() - start_merge
        log.info(f"Merge time for task {current_task_num}: {merge_time:.2f}s")

        # Evaluation (Identical to continual merging)
        eval_metrics: Dict[str, float] = {}
        start_eval = time.time()

        combined_test_sets = [test_sets[i] for i in range(current_task_num)]
        combined_test = ConcatDataset(combined_test_sets)
        combined_test_loader = DataLoader(
            combined_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        combined_metrics = eval_model(merged_model, combined_test_loader)
        for name, val in combined_metrics.items():
            eval_metrics[f"combined_{name}"] = val

        for pos, idx in enumerate(seen_task_indices):
            country = countries[idx]
            test_loader = DataLoader(
                test_sets[pos],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            country_metrics = eval_model(merged_model, test_loader)
            for name, val in country_metrics.items():
                eval_metrics[f"{country}_{name}"] = val

        eval_time = time.time() - start_eval
        log.info(f"Evaluation time for task {current_task_num}: {eval_time:.2f}s")

        # Calculate mean metrics across all individual countries
        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items():
            if not k.startswith("combined_"):
                _, met = k.split("_", 1)
                sums[met] += v
                counts[met] += 1
        mean_metrics = {f"mean_{m}": sums[m] / counts[m] for m in sums}

        # Compile results for this task
        row = {
            "task": current_task_num,
            "countries": tuple(seen_countries),
            "merge_time_s": merge_time,
            "eval_time_s": eval_time,
            "num_tasks_merged": len(seen_task_indices),
        }
        row.update(eval_metrics)
        row.update(mean_metrics)
        results.append(row)

        log.info(
            f"Task {current_task_num} completed. Combined test accuracy: "
            f"{eval_metrics.get('combined_accuracy', 'N/A')}"
        )
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


# -------------------------------------------------------------------------
# Configuration and main execution functions
# -------------------------------------------------------------------------


def main_from_config(config_path: str) -> pd.DataFrame:
    """
    Load configuration from file and execute the merging experiment.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
    
    Returns:
        pd.DataFrame: Results of the merging experiment
    """
    with open(config_path, "r") as f:
        cfg = (
            yaml.safe_load(f)
            if config_path.endswith((".yml", ".yaml"))
            else json.load(f)
        )

    from importlib import import_module

    # Dynamically import the specified module
    model_module_name = cfg.get("model_module", "SpectralGPT")
    logging.info(f"Using model module: {model_module_name}")

    try:
        model_module = import_module(model_module_name)

        # Import everything from the module into the global namespace
        for name in dir(model_module):
            if not name.startswith("_"):
                globals()[name] = getattr(model_module, name)

        # Also make transforms available for baseline imports
        if hasattr(model_module, "train_transform"):
            import builtins

            builtins.train_transform = model_module.train_transform
            builtins.val_transform = model_module.val_transform

        logging.info(f"Successfully imported {model_module_name} module")

    except ImportError as e:
        logging.error(f"Failed to import {model_module_name}: {e}")
        raise

    params = cfg["params"]
    params["model_module"] = model_module_name

    # Dynamically set metrics function if specified
    if isinstance(params.get("metrics_fn"), str):
        mod, fn = params["metrics_fn"].rsplit(":", 1)
        params["metrics_fn"] = getattr(import_module(mod), fn)

    test_type = cfg["test_type"]
    return test_merging(test_type, params)  # Add return statement


#  Merging dispatcher
def test_merging(test_type, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Dispatcher function to call the appropriate merging strategy.
    
    Args:
        test_type: Type of merging method to use
        params: Configuration parameters
    
    Returns:
        pd.DataFrame: Results of the merging experiment
    """
    # Determine which merging approach to use based on config or params
    merging_approach = params.get("merging_approach", "from_scratch")

    if merging_approach == "continual":
        return test_continual_merging(test_type, params)
    elif merging_approach == "from_scratch":
        return test_merging_from_scratch(test_type, params)
    else:
        raise ValueError(
            f"Unknown merging_approach: {merging_approach}. Use 'continual' or 'from_scratch'"
        )


# Setup logging
def setup_logging():
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Logger initialized")


# setup logging and parse command line arguments
def main():
    """
    Main entry point for the script.
    
    Parses command line arguments and executes the merging experiment
    based on the provided configuration file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Config path: {args.config}")

    # <- simple dump of every line in the config into the log
    with open(args.config, "r", encoding="utf-8") as f:
        logging.info("Config file contents:\n%s", f.read())

    print(main_from_config(args.config))


# def main():
#     setup_logging()
#     # --- Temporary debug configuration ---
#     config = {
#         'test_type': 'LoRASoups',
#         'model_module': 'SpectralGPT',
#         'params': {
#             'merging_approach': 'continual',
#             'countries': ['Finland', 'Ireland', 'Serbia', 'Portugal'],
#             'permutation': [0, 1, 2, 3],
#             'subset_fraction': 0.1,
#             'train_samples': 32,
#             'test_samples': 16,
#             'seed': 42,
#             'batch_size': 16,
#             'num_workers': 4,
#             'epoch': 1,
#             'lr': 1e-4,
#             'memory_size': 16,
#             'include_snowy': False,
#             'include_cloudy': False,
#             'save_dir': './saved_models',
#             'weight_base_path': "/faststorage/continual_low_rank_adaptation_of_remote_sensing_foundation_models/SpectralGPT/saved_models/epoch20/task_tuning"
#         }
#     }
#     logging.info("Using temporary debug configuration.")

#     # Manually call the core logic from main_from_config
#     test_type = config["test_type"]
#     params = config["params"]
#     params["model_module"] = config["model_module"]

#     results = test_merging(test_type, params)
#     print(results)


if __name__ == "__main__":
    main()
