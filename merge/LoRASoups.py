import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from typing import Iterable, List, Dict, Tuple, Optional, Any
import logging
from itertools import zip_longest
import types
from tqdm import tqdm

# Setup logger
log = logging.getLogger(__name__)


class _LoRaSoupParam(nn.Module):
    """
    A torch.nn.utils.parametrize-compatible module that adds a weighted
    sum of precomputed LoRA deltas to a weight matrix.

    The forward pass computes: W' = W + sum_h alpha[h] * deltas[h]

    Attributes:
        deltas (torch.Tensor): A buffer of shape [H, out_features, in_features]
                               containing the stacked LoRA delta weights.
        alpha (torch.Tensor or nn.Parameter): A vector of shape [H] for the
                                              head weights. Can be learnable
                                              or fixed.
    """
    def __init__(self,
                 deltas: List[torch.Tensor],
                 mode: str = "learnable"):
        super().__init__()
        if not deltas:
            raise ValueError("Deltas list cannot be empty.")
        
        # Stack H heads → [H, out, in]
        self.register_buffer('deltas', torch.stack(deltas))
        H = self.deltas.size(0)

        if mode == "learnable":
            # Initialize alpha_h = 1/H, making it a trainable parameter
            self.alpha = nn.Parameter(torch.full((H,), 1.0 / H))
        elif mode == "static":
            # Use a fixed, non-trainable buffer for alpha
            self.register_buffer('alpha', torch.full((H,), 1.0 / H))
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'learnable' or 'static'.")

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """Applies the weighted LoRA deltas to the base weight tensor."""
        # delta = sum_h alpha_h · deltas[h]
        delta = torch.einsum('h,hod->od', self.alpha, self.deltas)
        return W + delta


def _split_weight_bias(
    head: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extracts the weight and optional bias from a state dict."""
    try:
        weight = next(v for v in head.values() if v.ndim == 2)
        bias = next((v for v in head.values() if v.ndim == 1), None)
        return weight, bias
    except StopIteration:
        raise ValueError("Could not find a 2D weight tensor in the provided head.")


def _compute_deltas_for_layer(
    lid: str, lora_heads: List[Dict[str, torch.Tensor]], device: torch.device
) -> List[torch.Tensor]:
    """
    Computes the delta matrices for a specific layer across all heads.
    This function is robust to two representations:
    1. A pre-computed delta matrix (delta_{lid}).
    2. Decomposed matrices (w_a_{lid}, w_b_{lid}).
    """
    deltas = []
    for i, head in enumerate(lora_heads):
        delta_key = f"delta_{lid}"
        wa_key = f"w_a_{lid}"
        wb_key = f"w_b_{lid}"

        if delta_key in head:
            # Strategy 1: Use the pre-computed delta directly.
            deltas.append(head[delta_key].to(device))
            log.debug(f"Using pre-computed delta for head {i}, layer {lid}")
        elif wa_key in head and wb_key in head:
            # Strategy 2: Compute delta from A and B matrices.
            A = head[wa_key].to(device)  # [r, in]
            B = head[wb_key].to(device)  # [out, r]
            deltas.append(B @ A)
            log.debug(f"Computing delta from A/B for head {i}, layer {lid}")
        else:
            log.warning(
                f"Head {i} for layer {lid} is missing '{delta_key}' or "
                f"('{wa_key}', '{wb_key}'). Skipping this head for this layer."
            )
    return deltas


def _are_ranks_consistent(lora_heads: List[Dict[str, torch.Tensor]]) -> bool:
    """Checks if all LoRA heads have consistent ranks across all layers."""
    if not lora_heads:
        return True

    # Robustly discover all layer IDs from all heads that have w_a matrices
    all_w_a_keys = {k for h in lora_heads for k in h if k.startswith("w_a_")}
    
    # If any head lacks w_a keys, we must use delta merging.
    for i, head in enumerate(lora_heads):
        if not any(k.startswith("w_a_") for k in head):
            log.info(f"Head {i} is in delta-only format. Will use delta-based merging.")
            return False

    layer_ids = sorted({k.split('_')[-1] for k in all_w_a_keys})
    if not layer_ids:
        log.info("No w_a/w_b layers found in any head. Assuming delta-based merging is intended.")
        return False

    for lid in layer_ids:
        ranks = []
        for head in lora_heads:
            # All heads are guaranteed to have w_a keys at this point
            ranks.append(head[f"w_a_{lid}"].shape[0])
        
        if len(set(ranks)) > 1:
            log.info(f"Inconsistent ranks found for layer {lid}: {ranks}. Will use delta-based merging.")
            return False
            
    log.info("All LoRA heads have consistent ranks. Will use Zip-LoRA style merging.")
    return True


def _merge_lora_weights(
    layer_ids: List[str],
    soup_params: List[_LoRaSoupParam],
    lora_heads: List[Dict[str, torch.Tensor]],
    ranks_are_consistent: bool
) -> Dict[str, torch.Tensor]:
    """
    Creates a new merged LoRA state dict from the trained soup parameters.

    Args:
        layer_ids (List[str]): List of layer IDs.
        soup_params (List[_LoRaSoupParam]): List of trained parametrizations.
        lora_heads (List[Dict[str, torch.Tensor]]): Original LoRA heads.
        ranks_are_consistent (bool): Strategy selector for merging.

    Returns:
        Dict[str, torch.Tensor]: The state dict for the new merged LoRA adapter.
    """
    merged_lora: Dict[str, torch.Tensor] = {}
    for lid, soup_param in zip(layer_ids, soup_params):
        with torch.no_grad():
            alpha = soup_param.alpha.detach().cpu()  # [H]
            
            if not ranks_are_consistent:
                # Strategy 1: Ranks differ. Merge deltas directly.
                deltas = soup_param.deltas.cpu()  # [H, o, d]
                merged_delta = torch.einsum('h,hod->od', alpha, deltas)
                merged_lora[f"delta_{lid}"] = merged_delta
            else:
                # Strategy 2: Ranks are consistent. Use Zip-LoRA style merging.
                A_mats = torch.stack([h[f"w_a_{lid}"] for h in lora_heads])  # [H, r, d]
                B_mats = torch.stack([h[f"w_b_{lid}"] for h in lora_heads])  # [H, o, r]
                
                H, r, d = A_mats.shape
                o = B_mats.shape[1]
                
                # Scale A matrices by alpha weights and flatten
                A_scaled = A_mats * alpha.view(H, 1, 1)
                A_tot = A_scaled.reshape(H * r, d)
                
                # Permute and flatten B matrices
                B_tot = B_mats.permute(1, 0, 2).reshape(o, H * r)
                
                # Sanity check reconstruction
                delta_ref = torch.einsum("hor,hrd->hod", B_mats, A_scaled).sum(0)
                if not torch.allclose(B_tot @ A_tot, delta_ref, atol=1e-6):
                    log.error(f"Reconstruction mismatch in layer {lid}. Merged weights may be incorrect.")

                merged_lora[f"w_a_{lid}"] = A_tot.cpu()
                merged_lora[f"w_b_{lid}"] = B_tot.cpu()
                
    return merged_lora


def _cleanup_and_bake_weights(model: nn.Module) -> None:
    """
    Removes all parametrizations from the model, baking the final weights.
    The monkey-patched forward method is intentionally NOT restored, so the
    model continues to use the new classifier head.

    Args:
        model (nn.Module): The model to clean up.
    """
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and parametrize.is_parametrized(mod):
            # This call computes the final weight and removes the parametrization
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)

    # # Restore original forward method if it was monkey-patched
    # # We DO NOT restore it, so the model uses the new classifier.
    # if hasattr(model, "_orig_forward"):
    #     model.forward = model._orig_forward
    #     delattr(model, "_orig_forward")


def LoraSoupsMerge(
    pl_model: nn.Module,
    train_loader_old: Iterable,
    train_loader_new: Iterable,
    lora_heads: List[Dict[str, torch.Tensor]],
    classifier_heads: Optional[List[Dict[str, torch.Tensor]]] = None,
    mode: str = "learnable",
    num_epochs: int = 1,
    lr: float = 1e-4,
    current_task: int = 2, # Add current_task parameter
) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Merges multiple LoRA adapters by learning per-head, per-layer coefficients (alpha).

    This function freezes the base model, attaches learnable parametrizations to each
    LoRA-targeted linear layer, and trains the alpha coefficients (and optionally a new
    classifier head) on a combined dataset.

    Args:
        pl_model (nn.Module): The base model (e.g., a Pytorch Lightning module).
        train_loader_old (Iterable): DataLoader for the old task/replay data.
        train_loader_new (Iterable): DataLoader for the new task data.
        lora_heads (List[Dict[str, torch.Tensor]]): A list of state dicts, one for each LoRA adapter.
        classifier_heads (Optional[List[Dict[str, torch.Tensor]]]): Optional list of classifier state dicts.
        mode (str): Merging mode. "learnable" for trainable alpha, "static" for uniform alpha =1/H.
        num_epochs (int): Number of epochs to train the merging coefficients.
        lr (float): Learning rate for the optimizer.

    Returns:
        Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            - The model with merged weights baked in.
            - The state dict of the new merged LoRA adapter.
            - The state dict of the new merged classifier head.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)
    pl_model.train()

    # --- Add model structure printout ---
    log.info("--- Model Structure ---")
    log.info(pl_model)
    log.info("-----------------------")

    # 1. Freeze base model and collect trainable parameters
    for p in pl_model.parameters():
        p.requires_grad_(False)
    trainables: List[nn.Parameter] = []

    # 1) Gather zero-padded IDs from the LoRA keys:
    # e.g. layer_ids = ["000", "007", "013", ...]
    layer_ids = sorted({
        k.split('_')[-1]
        for k in lora_heads[0]
        if k.startswith("w_a_")
    })

    # Fallback for the case where the first head is a delta-only merge result.
    if not layer_ids and lora_heads:
        log.warning("Initial layer ID discovery failed (no 'w_a_' keys). "
                    "Falling back to robust discovery from all heads.")
        layer_ids = sorted(list(set(
            k.split('_')[-1]
            for h in lora_heads for k in h
            if k.startswith("w_a_") or k.startswith("delta_")
        )))

    log.debug(f"Discovered layer IDs: {layer_ids}")
    ranks_are_consistent = _are_ranks_consistent(lora_heads)

    # 3. For each layer, compute deltas and attach parametrization
    soup_params: List[_LoRaSoupParam] = []
    patched_modules: List[str] = []
    # Only target q and v layers as specified
    suffixes = ("q", "v")

    # 2) For each ID, convert to a “clean” integer string:
    for lid in layer_ids:
        deltas = _compute_deltas_for_layer(lid, lora_heads, device)
        if not deltas:
            log.warning(f"No deltas computed for layer ID {lid}, skipping patch for this layer.")
            continue

        parametrization = _LoRaSoupParam(deltas, mode=mode).to(device)
        found_module_for_lid = False
        clean_id = str(int(lid))  # "000" → "0", "007" → "7"

        # 3) Look through every submodule:
        for name, mod in pl_model.named_modules():
            # Match only Linear layers *and* ensure the name contains the block ID
            # e.g. name = "model.blocks.7.attn.q" for ViT structure
            if (
                isinstance(mod, nn.Linear)
                and f".blocks.{clean_id}.attn." in name
                and name.rsplit('.', 1)[-1] in suffixes
            ):
                delta_shape = parametrization.deltas.shape[1:]
                if mod.weight.shape != delta_shape:
                    log.debug(f"Skipping patch for '{name}': shape mismatch. "
                              f"Weight: {mod.weight.shape}, Delta: {delta_shape}")
                    continue

                # This is one of the LoRA-targeted linears!
                log.info(f"Patching module '{name}' for layer ID {lid}")
                parametrize.register_parametrization(mod, "weight", parametrization, unsafe=False)
                patched_modules.append(name)
                found_module_for_lid = True

        if found_module_for_lid:
            soup_params.append(parametrization)
            if mode == "learnable":
                trainables.append(parametrization.alpha)

    # If no modules were patched at all, something is wrong with the name matching.
    if not patched_modules:
        log.warning(
            "LoraSoupsMerge: Failed to find any matching Linear layers to patch. "
            "This may be due to a mismatch in layer names or tensor shapes. "
            "Exiting"
        )
        exit(1)

    # 4. Configure classifier head using weighted averaging
    if classifier_heads and len(classifier_heads) >= 2:
        # --- Ensure correct recursive weighting for continual learning ---
        # In a continual loop, we expect two heads: the previously merged one and the new one.
        # The weighting should be based on the number of tasks seen so far.
        if len(classifier_heads) > 2:
            log.warning(f"Received {len(classifier_heads)} classifier heads. "
                        "For correct weighting, ensure only the previous merge result and the new head are passed. "
                        "Using the first as 'old' and the last as 'new'.")

        # Weight by number of tasks (e.g., for task 3, weights are 2/3 for old, 1/3 for new)
        w_old = (current_task - 1) / current_task
        w_new = 1 / current_task

        log.info(f"Classifier merge weights for task {current_task}: old={w_old:.3f}, new={w_new:.3f}")

        # Use the first head as the accumulated "old" state and the last as the "new" task's head.
        weight_old, bias_old = _split_weight_bias(classifier_heads[0])
        weight_new, bias_new = _split_weight_bias(classifier_heads[-1])

        merged_weight = w_old * weight_old + w_new * weight_new
        merged_bias = None
        if bias_old is not None and bias_new is not None:
            merged_bias = w_old * bias_old + w_new * bias_new

        num_classes, hidden_dim = merged_weight.shape
        clf = nn.Linear(hidden_dim, num_classes,
                               bias=merged_bias is not None).to(device)
        with torch.no_grad():
            clf.weight.copy_(merged_weight)
            if merged_bias is not None:
                clf.bias.copy_(merged_bias)
        
        pl_model.classifier = clf
        pl_model.classifier.requires_grad_(True)
        trainables.extend(pl_model.classifier.parameters())

        # Monkey-patch the forward method
        if not hasattr(pl_model, "_orig_forward"):
            pl_model._orig_forward = pl_model.forward
        
        def _forward_with_classifier(self, x: torch.Tensor) -> torch.Tensor:
            feats = self._orig_forward(x)
            return self.classifier(feats)
        pl_model.forward = types.MethodType(_forward_with_classifier, pl_model)

    # 5. Setup optimizer and training loop
    # --- START: Add trainable parameter logging ---
    lora_params = sum(p.alpha.numel() for p in soup_params if hasattr(p, 'alpha'))
    cls_params = 0
    if hasattr(pl_model, "classifier"):
        cls_params = sum(p.numel() for p in pl_model.classifier.parameters())
    
    total_params = sum(p.numel() for p in trainables)

    log.info(f"[LoraSoupsMerge] Trainable LoRA-merge params : {lora_params:,}")
    log.info(f"[LoraSoupsMerge] Trainable classifier params : {cls_params:,}")
    log.info(f"[LoraSoupsMerge] Total trainables            : {total_params:,}")
    # --- END: Add trainable parameter logging ---

    if not trainables:
        log.warning("No trainable parameters found. Skipping training.")
        num_epochs = 0
        
    optim = torch.optim.Adam(trainables, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(num_epochs):
        # ---  Use zip to automatically handle different dataloader lengths ---
        pbar = tqdm(zip(train_loader_old, train_loader_new),
                    desc=f"Epoch {ep+1}/{num_epochs}", unit="batch")
        for batch_old, batch_new in pbar:

            x_old, y_old = batch_old
            x_new, y_new = batch_new
            
            x = torch.cat([x_old, x_new]).to(device)
            y = torch.cat([y_old, y_new]).to(device).float()

            out = pl_model(x)
            loss = loss_fn(out, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

    # 6. Build merged weights from trained parameters
    # Only build LoRA weights if there are soup_params to process
    merged_lora = {}
    if soup_params:
        merged_lora = _merge_lora_weights(layer_ids, soup_params, lora_heads, ranks_are_consistent)
    
    merged_classifier = {}
    if hasattr(pl_model, "classifier") and isinstance(pl_model.classifier, nn.Linear):
        merged_classifier['weight'] = pl_model.classifier.weight.detach().cpu()
        if pl_model.classifier.bias is not None:
            merged_classifier['bias'] = pl_model.classifier.bias.detach().cpu()

    # 7. Clean up model by baking weights and restoring original state
    _cleanup_and_bake_weights(pl_model)

    return pl_model, merged_lora, merged_classifier
