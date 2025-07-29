

"""
LoRAHub rewritten for vision-transformer + SpectralGPT.
Dependencies
------------
•  torch, torchvision, datasets, nevergrad  (same as before)
•  SpectralGPT repo on your PYTHONPATH
"""

from __future__ import annotations

import torch.nn
from .load import load_peft_models, convert_peft_to_ViT

import copy
import os
import random
from functools import partial
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import nevergrad as ng
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType

# ----------------------------------------------------------------------------
#  0.  Generic helper
# ----------------------------------------------------------------------------
def _linear_merge(
    weights: Sequence[float],
    lora_heads: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Perform linear merging of multiple LoRA heads with given weights.
    
    Args:
        weights: List of weights for each LoRA head
        lora_heads: List of LoRA state dictionaries (A/B matrices)
    
    Returns:
        Merged LoRA state dictionary
    """
    out: Dict[str, torch.Tensor] = {}
    keys = lora_heads[0].keys()
    for i, head in enumerate(lora_heads):
        for k in keys:
            out[k] = out.get(k, 0) + weights[i] * head[k]
    return out

def default_get_loss(train_dataloader, model, batch_size):
    """
    Calculate the loss of the model on the training dataset.
    This function computes the average loss over all batches in the dataloader.
    
    Args:
        train_dataloader: DataLoader containing training data
        model: The model to evaluate
        batch_size: Size of each batch
    
    Returns:
        Average loss value as float
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loss = torch.tensor(0.0, device=device)
    train_count = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pbar = tqdm(train_dataloader, desc="Calculating loss", leave=False)
        for batch in pbar:
            train_count += 1
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
            elif isinstance(batch, list):
                batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            with torch.no_grad():
                # Unpack the (x, y) tuple from the loader
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    images, labels = batch
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels)
                    labels = labels.float()  # Convert to float for BCEWithLogitsLoss
                else:
                    raise ValueError("Batch should be a tuple of (images, labels)")
                
                logits = model(images)
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(logits, labels)
            current_loss = loss.detach().float()
            train_loss += current_loss
            pbar.set_postfix({"batch_loss": current_loss.item()})
    loss = train_loss.float()
    avg_loss = float(train_loss) / train_count
    # average loss over the number of examples
    return avg_loss


def get_score(weights, model, cache, classifier, dataloader, batch_size, get_loss, get_regular):
    """
    Calculate the optimization score for given LoRA weights.
    This function combines the model loss with regularization to form the objective function.
    
    Args:
        weights: Current weights for LoRA heads
        model: The base model with LoRA adapters
        cache: Dictionary containing LoRA state dictionaries
        classifier: Classification head
        dataloader: DataLoader for evaluation
        batch_size: Size of each batch
        get_loss: Function to compute loss
        get_regular: Function to compute regularization term
    
    Returns:
        Combined score (loss + regularization)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    print("================ Get score ==================")
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        print(f"Loading LoRA head {peft_model_id}")
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
    model = model.to(device)
    classifier = classifier.to(device)
    model_clf = torch.nn.Sequential(model, classifier)
    # minimize the metric
    print("================ Get loss ==================")
    loss = get_loss(dataloader, model_clf, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    
    return metric_val


def default_l1_regularization(weights):
    """
    Calculate L1 regularization term for the weights.
    This helps prevent overfitting by penalizing large weight values.
    
    Args:
        weights: List of LoRA weights
    
    Returns:
        L1 regularization value
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def get_final_weights(weights, lora_heads: List[Dict[str, torch.Tensor]], cache):
    """
    Generate the final merged LoRA state dictionary using optimized weights.
    
    Args:
        weights: Optimized weights for each LoRA head
        lora_heads: List of LoRA state dictionaries
        cache: Dictionary containing LoRA state dictionaries
    
    Returns:
        Final merged LoRA state dictionary
    """
    final_state_dict = {}
    keys = cache[0].keys()
    for i, peft_sd in enumerate(lora_heads):
        lora_state_dict = cache[i]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict

# ----------------------------------------------------------------------------
#  Optimization steps to call from main
# ----------------------------------------------------------------------------
def lora_hub_learn(
    train_loader: Iterable,
    lora_heads: List[Dict[str, torch.Tensor]],
    LoRA_ViT_model: torch.nn.Module,
    classifier: torch.nn.Module,
    max_steps: int = 10,
    batch_size: int = 16,
    seed: int = 42,
    get_loss=default_get_loss, 
    get_regular=default_l1_regularization,
):
    """
    Main function to learn optimal LoRA weight combinations using gradient-free optimization.
    
    This function implements the LoRAHub algorithm which:
    1. Takes multiple pre-trained LoRA adapters
    2. Uses Nevergrad to find optimal weight combinations
    3. Merges the LoRA adapters with learned weights
    4. Returns the final merged model
    
    Args:
        train_loader: DataLoader for training data
        lora_heads: List of LoRA state dictionaries to merge
        LoRA_ViT_model: Base Vision Transformer model
        classifier: Classification head
        max_steps: Maximum optimization steps
        batch_size: Batch size for evaluation
        seed: Random seed for reproducibility
        get_loss: Function to compute loss (default: default_get_loss)
        get_regular: Function to compute regularization (default: default_l1_regularization)
    
    Returns:
        Tuple of (merged_model, final_lora_state_dict)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Move models to device
    LoRA_ViT_model = LoRA_ViT_model.to(device)
    classifier = classifier.to(device)
    
    # Move LoRA heads to device
    lora_heads = [{k: v.to(device) for k, v in head.items()} for head in lora_heads]
    number_of_loras = len(lora_heads)

    # Provided: lora_heads
    # load_peft_models shall return: model, cache
    # model contains the current base_model + current LoRA adapters
    # cache contains the current base_model + different LoRA adapters
    peft_model, cache = load_peft_models(lora_heads, LoRA_ViT_model)


    # Create partial function for optimization objective
    get_score_partial = partial(get_score, 
                                model=peft_model, 
                                cache=cache,
                                classifier=classifier,
                                dataloader=train_loader,
                                batch_size=batch_size,
                                get_loss=get_loss, 
                                get_regular=get_regular)
    

    # Nevergrad optimization
    # Define parameter space for optimization
    instrum = ng.p.Array(
        init=[0.0] * number_of_loras,  # Initialize all weights to 0
        upper=[1.5] * number_of_loras,  # Upper bound for weights
        lower=[-1.5] * number_of_loras,  # Lower bound for weights
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_steps)
    print("> Begin to perform gradient-free optimization ...")
    optimizer.register_callback("tell", ng.callbacks.ProgressBar())
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    
    # Generate final merged LoRA state dictionary
    final_lora = get_final_weights(recommendation.value, lora_heads, cache)

    # ------------------------------------------------------------------
    # Finalize merged model
    # ------------------------------------------------------------------
    set_peft_model_state_dict(peft_model, final_lora)
    peft_model = peft_model.merge_and_unload()
    LoRA_ViT_model = convert_peft_to_ViT(peft_model, LoRA_ViT_model)

    return LoRA_ViT_model, final_lora

