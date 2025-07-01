import argparse
import json
import logging
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

# Import transforms along with other SpectralGPT components
from SpectralGPT import (
    eval_model,
    load_model,
    train_transform,  
    val_transform,    
    SpectralGPTLightningModule, 
)

from baseline import (
    get_datasets,
)

# -------------------------------------------------------------------------
#  Helper: randomly pick 'n' indices from a dataset
# -------------------------------------------------------------------------


def sample_stratified_subset(dataset, n, seed):
    """Return a stratified Subset of size min(n, len(dataset)) maintaining class proportions."""
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
            indices, 
            train_size=n, 
            stratify=labels, 
            random_state=seed
        )
    except ValueError:  # Fall back to random if stratification fails
        g = torch.Generator().manual_seed(seed)
        selected_indices = torch.randperm(len(dataset), generator=g)[:n].tolist()
        logging.warning(
            "Stratified sampling failed, falling back to random sampling."
        )
   
    return Subset(dataset, selected_indices)


# -------------------------------------------------------------------------
#  Helpers: load the weights
# -------------------------------------------------------------------------


def load_lora_weights(country: str, base_path: str = None):
    """Load LoRA weights for a specific country."""
    try:
        from safetensors.torch import load_file
        
        if base_path is None:
            base_path = "/faststorage/continual_low_rank_adaptation_of_remote_sensing_foundation_models/SpectralGPT/task_tuning/saved_models"
        
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
    """Load classifier weights for a specific country."""
    try:
        from safetensors.torch import load_file
        
        if base_path is None:
            base_path = "/faststorage/continual_low_rank_adaptation_of_remote_sensing_foundation_models/SpectralGPT/task_tuning/saved_models"
        
        fc_path = os.path.join(base_path, f"{country}_fc.safetensors")
        
        if os.path.exists(fc_path):
            logging.info(f"Loading classifier weights from: {fc_path}")
            return load_file(fc_path)
        else:
            logging.warning(f"Classifier weights not found: {fc_path}")
            return None
            
    except (FileNotFoundError, ImportError) as e:
        logging.warning(f"Could not load classifier weights for {country}: {e}")
        return None


def load_weights_for_permutation(countries, permutation, base_path: str = None):
    """Load weights for countries according to permutation order."""
    all_lora_weights = []
    all_classifier_weights = []
    
    # Load weights according to permutation order
    for country_idx in permutation:
        if country_idx < len(countries):
            country = countries[country_idx]
            
            logging.info(f"Loading weights for country: {country} (permutation index: {country_idx})")
            
            lora_weights = load_lora_weights(country, base_path)
            classifier_weights = load_classifier_weights(country, base_path)
            
            if lora_weights is not None:
                all_lora_weights.append(lora_weights)
            else:
                logging.warning(f"No LoRA weights found for {country}")
                
            if classifier_weights is not None:
                all_classifier_weights.append(classifier_weights)
            else:
                logging.warning(f"No classifier weights found for {country}")
    
    return all_lora_weights, all_classifier_weights

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
    subset_fraction = params.get("subset_fraction", 0.1)
    seed = params.get("seed", 0)
    batch_size = params.get("batch_size", 16)
    num_workers = params.get("num_workers", 4)
    epoch = params.get("epoch", 1)
    lr = float(params.get("lr", 1e-4))
    model_module = params.get("model_module")
    
    # Load full datasets first
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

    # Create stratified subsets for every original task
    train_subsets = []
    val_subsets = []
    for i, (train_set, val_set) in enumerate(zip(full_train_sets, full_val_sets)):
        train_subset_size = max(1, int(len(train_set) * subset_fraction))
        val_subset_size = max(1, int(len(val_set) * subset_fraction))

        train_subsets.append(sample_stratified_subset(train_set, train_subset_size, seed + i))
        val_subsets.append(sample_stratified_subset(val_set, val_subset_size, seed + i))

        log.info(
            f"Task {i + 1} subset sizes: train={train_subset_size}/{len(train_set)}, "
            f"val={val_subset_size}/{len(val_set)}"
        )

    save_dir = params.get("save_dir")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load base model without LoRA wrapping
    if model_module == "SpectralGPT":
        from SpectralGPT import SpectralGPTLightningModule
        base_model = load_model(r=4, use_lora=False)
    elif model_module == "SoftCon":
        from SoftCon import SoftConLightningModule  # Adjust import as needed
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

    for current_task in range(2, len(permutation) + 1):
        seen_task_indices = permutation[:current_task]
        seen_countries = [countries[i] for i in seen_task_indices]
        log.info(f"Task {current_task}: Merging countries {seen_countries}")

        # Pool the task data and create train/validation sets
        pooled_train_sets = [train_subsets[i] for i in seen_task_indices]
        pooled_val_sets = [val_subsets[i] for i in seen_task_indices]

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
            pl_model = SpectralGPTLightningModule(base_model, num_classes=19, lr=lr)
        elif model_module == "SoftCon":
            pl_model = SoftConLightningModule(base_model, num_classes=19, lr=lr)
        else:
            raise ValueError(f"Unknown model_module: {model_module}")

        # Apply specific merging strategy
        start_merge = time.time()

        # Get weights for current tasks (permutation-aware)
        current_lora_weights = all_lora_weights[:current_task]
        current_classifier_weights = all_classifier_weights[:current_task]

        if test_type == "LoRASoups":
            from LoRASoups import LoRASoupsMerge

            merged_model, merged_lora, merged_classifier = LoRASoupsMerge(
                pl_model=pl_model,
                train_loader=train_loader,
                val_loader=val_loader,
                lora_heads=current_lora_weights,      # List of LoRA weights
                classifier_heads=current_classifier_weights,  # List of classifier weights
                mode='learnable',                     # or 'static'
                num_epochs=epoch,
                lr=lr
            )
        elif test_type == "ZipLoRA":
            # Add ZipLoRA implementation
            pass
        elif test_type == "LoRAHub":
            # Add LoRAHub implementation
            pass
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        merge_time = time.time() - start_merge
        log.info(f"Merge time for task {current_task}: {merge_time:.2f}s")

        # Save merged model
        if save_dir:
            path = os.path.join(
                save_dir, f"merged_task{current_task}-{'-'.join(seen_countries)}.pkl"
            )
            joblib.dump(
                {
                    "model": merged_model,
                    "lora_weights": merged_lora,
                    "classifier_weights": merged_classifier,
                },
                path,
            )
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

        log.info(
            f"Task {current_task} completed. Combined test accuracy: "
            f"{eval_metrics.get('combined_accuracy', 'N/A')}"
        )

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
#         model_module=model_module,
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
#         model_module=model_module,
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
#         # TODO we need to load the 2 models correc



# ---------------------------------------------------------
#
# ---------------------------------------------------------


def main_from_config(config_path: str) -> pd.DataFrame:
    """Load config and run the appropriate test based on test_type."""
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
        globals().update(
            {
                name: getattr(model_module, name)
                for name in dir(model_module)
                if not name.startswith("_")
            }
        )
        
        # CRITICAL: Make transforms available to baseline.py
        import baseline
        if hasattr(model_module, 'train_transform'):
            baseline.train_transform = model_module.train_transform
            baseline.val_transform = model_module.val_transform
            logging.info("Successfully set transforms in baseline module")
        else:
            logging.error("train_transform not found in model module")
            
    except ImportError as e:
        logging.error(f"Could not import {model_module_name}: {e}")
        # Fallback approach
        try:
            from SpectralGPT import train_transform, val_transform
            import baseline
            baseline.train_transform = train_transform
            baseline.val_transform = val_transform
            logging.info("Fallback: Set transforms via direct SpectralGPT import")
        except ImportError as e2:
            logging.error(f"Could not import transforms at all: {e2}")
            raise

    params = cfg["params"]
    
    # Add model_module to params
    params["model_module"] = model_module_name

    # Dynamically set metrics function if specified
    if isinstance(params.get("metrics_fn"), str):
        mod, fn = params["metrics_fn"].rsplit(":", 1)
        params["metrics_fn"] = getattr(import_module(mod), fn)

    test_type = cfg["test_type"]
    
    # Handle different test types
    if test_type in ["LoRASoups", "ZipLoRA", "LoRAHub"]:
        # New merging test types
        logging.info(f"Running merging experiment: {test_type}")
        return test_merging(test_type, params)
    
    elif test_type in ["replay", "no_replay", "task_tuning"]:
        # Original test types
        model = load_model(4)
        if test_type == "replay":
            return test_finetuning_from_scratch(model, params)
        elif test_type == "no_replay":
            return test_continual_finetuning(model, params)
        elif test_type == "task_tuning":
            return test_task_tuning(model, params)
    else:
        raise ValueError(f"Unknown test_type: {test_type}. Supported: LoRASoups, ZipLoRA, LoRAHub, replay, no_replay, task_tuning")


def setup_logging():
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.info("Logger initialized")


def main():
    """Main entry point - parse arguments and run experiment."""
    parser = argparse.ArgumentParser(description="Run continual learning experiments")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Starting experiment with config: {args.config}")

    # Log the config file contents for debugging
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config_contents = f.read()
            logging.info("Config file contents:\n%s", config_contents)
    except Exception as e:
        logging.error(f"Could not read config file: {e}")
        return

    # Run the experiment
    try:
        results_df = main_from_config(args.config)
        
        # Save results
        if results_df is not None and not results_df.empty:
            output_path = args.config.replace('.yml', '_results.csv').replace('.yaml', '_results.csv')
            results_df.to_csv(output_path, index=False)
            logging.info(f"Results saved to: {output_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("EXPERIMENT RESULTS")
            print("="*60)
            print(results_df.to_string(index=False))
            print("="*60)
        else:
            logging.warning("No results returned from experiment")
            
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()