"""
load.py
LoRA loading and conversion utilities for Vision Transformer models.

This module provides functions to:
1. Load and validate LoRA adapters for Vision Transformers
2. Convert between different model formats (PEFT, custom LoRA ViT)
3. Check consistency between base models and LoRA adapters
4. Extract and merge LoRA state dictionaries

Dependencies
------------
• torch, peft, logging
• Custom ViT model from baseline/model/video_vit
"""

from __future__ import annotations
import copy
from typing import Dict, List
import re
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, TaskType
import sys
import os
import torch
import tqdm
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../baseline")
from model.video_vit import vit_base_patch8_128 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_base_sd_consistency(peft_model, base_sd, verbose=True):
    """
    Check if the base layers in peft_model match with base_sd.
    This function automatically handles the additional '.base_layer' wrapper in PEFT models.
    
    Args:
        peft_model: PEFT model to check
        base_sd: Base model state dictionary
        verbose: If True, print detailed comparison information
    
    Returns:
        bool: True if all base weights match, False otherwise
    """
    model_sd = peft_model.state_dict()
    all_match = True
    checked_count = 0
    mismatch_count = 0

    # LoRA-related layers to ignore during base model comparison
    lora_keywords = {
        'lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B',
        'lora_dropout', 'lora_magnitude_vector'
    }
    def is_lora_param(key: str) -> bool:
        return any(word in key for word in lora_keywords)

    prefix = "base_model.model."

    def make_peft_candidates(base_key: str):
        """
        Generate all possible keys that might appear in the peft model for a given base key.
        Handles the different naming conventions between base models and PEFT models.
        """
        cand = [f"{prefix}{base_key}"]  # ① Add prefix directly
        # ② Insert .base_layer before the last level
        #    e.g., attn.q.bias -> attn.q.base_layer.bias
        if re.search(r'\.(weight|bias)$', base_key):
            base_layer_key = re.sub(r'\.(weight|bias)$', r'.base_layer.\1', base_key)
            cand.append(f"{prefix}{base_layer_key}")
        return cand

    for k, v in base_sd.items():
        if is_lora_param(k):
            continue  # Skip LoRA parameters

        # Find the first matching candidate key in the list
        peft_key = next((ck for ck in make_peft_candidates(k) if ck in model_sd), None)
        if peft_key is None:
            if k == "fc.weight" or k == "fc.bias":
                continue
            if verbose:
                logger.warning(f"[Missing] {k} ↛ (Can't find in peft_model)")
            all_match = False
            mismatch_count += 1
            continue

        # Compare weights
        try:
            if not torch.allclose(model_sd[peft_key], v, atol=1e-6):
                if verbose:
                    logger.warning(f"[Mismatch] {k} ≠ {peft_key}")
                    logger.warning(f"  base_sd  shape: {v.shape}")
                    logger.warning(f"  peft_sd  shape: {model_sd[peft_key].shape}")
                    logger.warning(f"  max |Δ|  : {(model_sd[peft_key] - v).abs().max().item():.3e}")
                all_match = False
                mismatch_count += 1
            checked_count += 1
        except Exception as e:
            logger.error(f"[Error] Error comparing {k} : {e}")
            all_match = False
            mismatch_count += 1

    # === Summary ===
    logger.info("\n=== Summary ===")
    logger.info(f"Total parameters checked : {checked_count}")
    logger.info(f"Total mismatches/missing : {mismatch_count}")
    if all_match:
        logger.info("✅ All base weights loaded correctly!")
    else:
        logger.warning("❌ Uncorrect base weights loaded. See above log.")

    return all_match


def check_peft_sd_consistency(peft_model, peft_sd, verbose=True):
    """
    Check if the LoRA weights in peft_model match with peft_sd.
    This function validates that LoRA adapters are correctly loaded.

    Args:
        peft_model: Model with LoRA/PEFT loaded (nn.Module)
        peft_sd: LoRA/PEFT state_dict (usually loaded from adapter_model.bin)
        verbose: If True, print detailed comparison information
    
    Returns:
        bool: True if all LoRA weights match, False otherwise
    """
    model_sd = peft_model.state_dict()
    all_match      = True
    checked_count  = 0
    mismatch_count = 0

    # Only focus on LoRA-related keywords (can be adjusted as needed)
    lora_keywords = { "lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B",
                      "lora_dropout", "lora_magnitude_vector" }

    def is_lora_param(key: str) -> bool:
        return any(w in key for w in lora_keywords)

    # -------------- Key: Generate all possible keys that might appear in peft_model.state_dict() -----------------
    model_prefixes = ["", "base_model.model."]              # Whether to include base prefix
    def insert_default(seg: str) -> str:                    # Convert .lora_X.weight -> .lora_X.default.weight
        return re.sub(r"(lora_[A-Z]\w*)\.(\w+)$", r"\1.default.\2", seg)

    def make_candidates(k: str):
        """
        Generate candidate keys for LoRA parameters in PEFT models.
        Handles different naming conventions and prefixes.
        """
        cands = []
        # ① As is / with prefix
        for pre in model_prefixes:
            cands.append(pre + k)
        # ② Insert .default
        k_default = insert_default(k)
        if k_default != k:
            for pre in model_prefixes:
                cands.append(pre + k_default)
        return cands
    # ---------------------------------------------------------------------

    for k, v in peft_sd.items():
        if not is_lora_param(k):
            # Sometimes adapter_model.bin contains non-LoRA parameters (e.g., alpha/gate)
            # You can decide whether to ignore them based on your needs
            continue

        # Get the first candidate key that actually exists in model_sd
        peft_key = next((ck for ck in make_candidates(k) if ck in model_sd), None)

        if peft_key is None:
            # No corresponding weight found in the model
            if verbose:
                logger.warning(f"[Missing] {k} ↛ (Can't find in peft_model)")
            all_match = False
            mismatch_count += 1
            continue

        # Compare values
        try:
            if not torch.allclose(model_sd[peft_key], v, atol=1e-6):
                if verbose:
                    logger.warning(f"[Mismatch] {k} ≠ {peft_key}")
                    logger.warning(f"  peft_sd shape: {v.shape}")
                    logger.warning(f"  model_sd shape: {model_sd[peft_key].shape}")
                    logger.warning(f"  max |Δ|       : {(model_sd[peft_key] - v).abs().max().item():.3e}")
                all_match = False
                mismatch_count += 1
            checked_count += 1
        except Exception as e:
            logger.error(f"[Error] Error comparing {k}: {e}")
            all_match = False
            mismatch_count += 1

    # ----------- Summary -----------
    logger.info("\n=== PEFT Summary ===")
    logger.info(f"Total parameters checked : {checked_count}")
    logger.info(f"Total mismatches/missing : {mismatch_count}")
    if all_match:
        logger.info("✅ All lora LoRA keys match！")
    else:
        logger.warning("❌ LoRA keys missing or mismatch, see above log.")

    return all_match

def extract_model_info(LoRA_ViT_model: torch.nn.Module):
    """
    Extract and print model information from LoRA_ViT_model.
    This function provides detailed information about the model structure and parameters.
    
    Args:
        LoRA_ViT_model: The LoRA Vision Transformer model to analyze
    
    Returns:
        tuple: (pytorch_model, pytorch_state_dict)
    """
    # Get model inside
    pytorch_model = LoRA_ViT_model
    pytorch_state_dict = pytorch_model.state_dict()
    
    logger.info("\n=== PyTorch Model Info ===")
    logger.info(f"Type: {type(pytorch_model)}")
    logger.info(f"Total parameters: {sum(p.numel() for p in pytorch_model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)}")
    
    logger.info("\n=== State Dict Keys ===")
    logger.info(f"PyTorch model keys: {list(pytorch_state_dict.keys())[:10]} ...")
    
    return pytorch_model, pytorch_state_dict


def extract_sd(LoRA_ViT_model):
    """
    Extract LoRA and base state dictionaries from a custom LoRA ViT model.
    This function separates LoRA parameters (w_a, w_b) from base model parameters.
    
    Args:
        LoRA_ViT_model: Custom LoRA Vision Transformer model
    
    Returns:
        tuple: (peft_sd, base_sd) - LoRA and base state dictionaries
    """
    peft_sd, base_sd = {}, {}
    for k, v in LoRA_ViT_model.state_dict().items():
        logger.debug(k)  # Changed to debug level since this is detailed debugging info
        if ".w_a." in k:
            # Convert custom LoRA A weights to PEFT format
            new_k = re.sub(r"\.w_a\.weight$", ".lora_A.weight", k.replace("lora_vit.", ""))
            peft_sd[new_k] = v
        elif ".w_b." in k:
            # Convert custom LoRA B weights to PEFT format
            new_k = re.sub(r"\.w_b\.weight$", ".lora_B.weight", k.replace("lora_vit.", ""))
            peft_sd[new_k] = v
        elif ".w.weight" in k:
            # Extract base model weights
            new_k = k.replace("lora_vit.", "").replace(".w.weight", ".weight")
            base_sd[new_k] = v
        elif ".w.bias" in k:
            # Extract base model biases
            new_k = k.replace("lora_vit.", "").replace(".w.bias", ".bias")
            base_sd[new_k] = v
        else:                   
            # Other parameters (non-LoRA)
            base_sd[k.replace("lora_vit.", "")  ] = v
    logger.debug(f"PEFT state dict keys: {peft_sd.keys()}")  # Changed to debug level
    logger.debug(f"Base state dict keys: {base_sd.keys()}")  # Changed to debug level
    
    return peft_sd, base_sd
            
def load_peft_model(base_sd, lora_head):
    """
    Load a PEFT model with given base and LoRA state dictionaries.
    This function creates a PEFT model from scratch and validates the loading.
    
    Args:
        base_sd: Base model state dictionary
        lora_head: LoRA adapter state dictionary
    
    Returns:
        tuple: (lora_head, peft_model) - Loaded LoRA state dict and PEFT model
    
    Raises:
        ValueError: If base or LoRA state dictionaries are inconsistent
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create base ViT model
    base = vit_base_patch8_128(sep_pos_embed=True, num_classes=19)          # TODO: import SVit
    base = base.to(device)
    base_sd = {k: v.to(device) for k, v in base_sd.items()}
    missing_keys = base.load_state_dict(base_sd, strict=False)
    logger.info("Missing keys:", missing_keys)
    
    # Configure LoRA settings
    cfg = LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.0,
        target_modules=["attn.q", "attn.v"],
        bias="none",
        inference_mode=True
    )
    # Get peft model
    peft_model = get_peft_model(base, cfg)

    # Load LoRA weights into PEFT model
    peft_model = peft_model.to(device)
    lora_head = {k: v.to(device) for k, v in lora_head.items()}
    missing, unexpected = set_peft_model_state_dict(peft_model, lora_head)
    # Keep these print statements since they're near a raise statement
    print("PEFT load missing:", missing)
    print("PEFT load unexpected:", unexpected)
    
    # Validate loading consistency
    if not check_base_sd_consistency(peft_model, base_sd):
        raise ValueError("Base SD is not consistent with PEFT model")
    if not check_peft_sd_consistency(peft_model, lora_head):
        raise ValueError("PEFT SD is not consistent with PEFT model")
    return lora_head, peft_model



def load_peft_models(lora_heads: List[Dict[str, torch.Tensor]], LoRA_ViT_model: torch.nn.Module):
    """
    Load multiple LoRA adapters and create a cache of PEFT models.
    This function is the main entry point for loading multiple LoRA adapters.
    
    Args:
        lora_heads: List of LoRA state dictionaries to load
        LoRA_ViT_model: Base LoRA Vision Transformer model
    
    Returns:
        tuple: (peft_model, cache) - Main PEFT model and cache of all LoRA adapters
    
    Raises:
        Exception: If LoRA adapters have incompatible architectures
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LoRA_ViT_model = LoRA_ViT_model.to(device)
    
    # Extract model information and state dictionaries
    extract_model_info(LoRA_ViT_model)
    LoRA_ViT_model.eval()
    peft_sd, base_sd = extract_sd(LoRA_ViT_model)
    peft_sd_prefixed, peft_model = load_peft_model(base_sd, peft_sd)
    
    # Create cache of all LoRA adapters
    cache = {}
    first_dict = None
    for idx, lora_head in enumerate(lora_heads):
        logger.info(f"> Loading LoRA head {idx}")
        cur_peft_sd, cur_peft_model = load_peft_model(base_sd, lora_head)
        cache[idx] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))
        if first_dict is None:
            first_dict = cache[idx]
        # Check whether the LoRA can be merged into one 
        try:
            # Detect whether the architecture is the same across all LoRA adapters
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[idx][key].shape
        except:
            raise Exception(f'LoRA Modules {idx} cannot be merged since it has a different arch (e.g., rank).')
    return peft_model, cache


def convert_peft_to_ViT(peft_model, LoRA_ViT_model):
    """
    Convert a PEFT model back to the custom LoRA ViT format.
    This function remaps PEFT state dictionary keys to the custom LoRA ViT format.
    
    Args:
        peft_model: PEFT model to convert
        LoRA_ViT_model: Target custom LoRA ViT model
    
    Returns:
        LoRA_ViT_model: Updated custom LoRA ViT model
    
    Raises:
        ValueError: If there are missing or unexpected keys during conversion
    """
    def remap_key(k: str) -> str:
        """
        Turn a `base_model.model.*` key into its `lora_vit.*` twin.
        This function handles the key mapping between PEFT and custom LoRA ViT formats.
        """
        k = k.replace("base_model.model.", "lora_vit.")    # prefix
        k = k.replace(".w.base_layer.", ".w.")             # fused backbone weight / bias
        k = k.replace(".w.lora_A.default.", ".w_a.")       # LoRA A  →  w_a
        k = k.replace(".w.lora_B.default.", ".w_b.")       # LoRA B  →  w_b
        return k
    
    peft_sd = peft_model.state_dict()
    peft_sd_renamed = {remap_key(k): v for k, v in peft_sd.items()}
    missing, unexpected = LoRA_ViT_model.load_state_dict(peft_sd_renamed, strict=False)
    
    # Validate conversion
    if len(missing) > 0:
        raise ValueError("Missing keys in LoRA_ViT_model during converting peft to ViT:", missing)
    if len(unexpected) > 0:
        raise ValueError("Unexpected keys in LoRA_ViT_model during converting peft to ViT:", unexpected)
    return LoRA_ViT_model