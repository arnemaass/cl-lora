import logging
import types
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from tqdm import tqdm
from transformers.data.data_collator import default_data_collator
from .algorithms import lora_hub_learn  
import torch
from torch.utils.data import Subset, DataLoader


# Setup logger
log = logging.getLogger(__name__)

def _split_weight_bias(
    head: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extracts the weight and optional bias from a state dict."""
    try:
        weight = next(v for v in head.values() if v.ndim == 2)
        bias = next((v for v in head.values() if v.ndim == 1), None)
        return weight, bias
    except StopIteration:
        raise ValueError("Could not find a 2D weight tensor in the provided head.")


def _cleanup_and_bake_weights(model: nn.Module) -> None:
    """
    Removes all parametrizations from the model, baking the final weights.
    The monkey-patched forward method is intentionally NOT restored, so the
    model continues to use the new classifier head.

    Args:
        model (nn.Module): The model to clean up.
    """
    # 1) Remove all parametrizations (bakes deltas into weights)
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and parametrize.is_parametrized(mod):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)

    # 2) If a classifier was added, re-install a fresh eval-forward
    if hasattr(model, "_orig_forward") and hasattr(model, "classifier"):
        # Capture the original forward method in a local variable for the closure.
        original_forward = model._orig_forward

        def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
            # Call the captured original forward method, not via self.
            feats = original_forward(x)  # encoder output
            return self.classifier(feats)  # baked-in classifier

        model.forward = types.MethodType(eval_forward, model)
        # Now it's safe to delete the attribute.
        delattr(model, "_orig_forward")

def LoraHubMerge_continual(
    pl_model: nn.Module,
    train_loader_old, # all train data of old tasks and curr task
    train_loader_new, # only curr task data
    lora_heads: List[Dict[str, torch.Tensor]],
    classifier_heads: Optional[List[Dict[str, torch.Tensor]]] = None,
    mode: str = "learnable",
    num_epochs: int = 1,
    lr: float = 1e-4,
    current_task: int = 2,  # Add current_task parameter
) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Incrementally merges a new task's LoRA adapter into a previously merged adapter.

    This function is designed for a continual learning scenario. It takes the result
    of the previous merge (one LoRA adapter, one classifier) and the components of a
    new task, then creates a new merged state.

    - **LoRA Merging**: Learns coefficients to combine the previous merged LoRA adapter
      and the new task's LoRA adapter.
    - **Classifier Merging**: Applies a weighted average to the classifier heads based
      on the task count (e.g., for task T, old_weight=(T-1)/T, new_weight=1/T).

    Args:
        pl_model (nn.Module): The base model (e.g., a Pytorch Lightning module).
        train_loader_old (Iterable): DataLoader for the old task/replay data.
        train_loader_new (Iterable): DataLoader for the new task data.
        lora_heads (List[Dict[str, torch.Tensor]]): A list of 2 LoRA state dicts:
            [0]: The previously merged LoRA adapter.
            [1]: The new task's LoRA adapter.
        classifier_heads (Optional[List[Dict[str, torch.Tensor]]]): A list of 2 classifier state dicts:
            [0]: The previously merged classifier head.
            [1]: The new task's classifier head.
        mode (str): Merging mode. "learnable" for trainable alpha, "static" for uniform alpha.
        num_epochs (int): Number of epochs to train the merging coefficients.
        lr (float): Learning rate for the optimizer.
        current_task (int): The number of the current task (e.g., 2, 3, ...). Used for weighting.

    Returns:
        Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            - The model with the newly merged weights baked in.
            - The state dict of the new merged LoRA adapter.
            - The state dict of the new merged classifier head.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)
    pl_model.train()

    # --- Add model structure printout, only for the first merge task ---
    if current_task == 2:
        log.info("--- Model Structure ---")
        log.info(pl_model)
        log.info("-----------------------")

    full_ds = train_loader_old.dataset 
    n_total = len(full_ds)
    k = n_total // 10 
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=g)[:k]
    sub_ds = Subset(full_ds, indices.tolist())
    hubloader = DataLoader(
        sub_ds,
        batch_size=train_loader_old.batch_size,
        shuffle=False,
        num_workers=train_loader_old.num_workers,   
        pin_memory=True,
    )
    # 1. Freeze base model and collect trainable parameters
    for p in pl_model.parameters():
        p.requires_grad_(False)
    trainables: List[nn.Parameter] = []

    # 2. Gather zero-padded IDs from the LoRA keys:
    # e.g. layer_ids = ["000", "007", "013", ...]
    layer_ids = sorted(
        {k.split("_")[-1] for k in lora_heads[0] if k.startswith("w_a_")}
    )

    # Fallback for the case where the first head is a delta-only merge result.
    if not layer_ids and lora_heads:
        log.warning(
            "Initial layer ID discovery failed (no 'w_a_' keys). "
            "Falling back to robust discovery from all heads."
        )
        layer_ids = sorted(
            list(
                set(
                    k.split("_")[-1]
                    for h in lora_heads
                    for k in h
                    if k.startswith("w_a_") or k.startswith("delta_")
                )
            )
        )

    log.debug(f"Discovered layer IDs: {layer_ids}")

    # 3. For each layer, compute deltas and attach parametrization
    # Only target q and v layers (since these have been trained)
    suffixes = ("q", "k")


    # 4. Configure classifier head using weighted averaging
    if classifier_heads and len(classifier_heads) >= 2:
        # --- Ensure correct recursive weighting for continual learning ---
        # In a continual loop, we expect two heads: the previously merged one and the new one.
        # The weighting should be based on the number of tasks seen so far.
        if len(classifier_heads) > 2:
            log.warning(
                f"Received {len(classifier_heads)} classifier heads. "
                "For correct weighting, ensure only the previous merge result and the new head are passed. "
                "Using the first as 'old' and the last as 'new'."
            )

        # Weight by number of tasks (e.g., for task 3, weights are 2/3 for old, 1/3 for new)
        w_old = (current_task - 1) / current_task
        w_new = 1 / current_task

        log.info(
            f"Classifier merge weights for task {current_task}: old={w_old:.3f}, new={w_new:.3f}"
        )

        # Use the first head as the accumulated "old" state and the last as the "new" task's head.
        weight_old, bias_old = _split_weight_bias(classifier_heads[0])
        weight_new, bias_new = _split_weight_bias(classifier_heads[-1])

        merged_weight = w_old * weight_old + w_new * weight_new
        merged_bias = None
        if bias_old is not None and bias_new is not None:
            merged_bias = w_old * bias_old + w_new * bias_new

        num_classes, hidden_dim = merged_weight.shape
        clf = nn.Linear(hidden_dim, num_classes, bias=merged_bias is not None).to(
            device
        )
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
    cls_params = 0
    if hasattr(pl_model, "classifier"):
        cls_params = sum(p.numel() for p in pl_model.classifier.parameters())

    total_params = sum(p.numel() for p in trainables)

    log.info(f"[LoraHubMerge] Trainable classifier params : {cls_params:,}")
    log.info(f"[LoraHubMerge] Total trainables            : {total_params:,}")
    # --- END: Add trainable parameter logging ---

    if not trainables:
        log.warning("No trainable parameters found. Skipping training.")
        num_epochs = 0

    optim = torch.optim.Adam(trainables, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(num_epochs):
        # ---  Use zip to automatically handle different dataloader lengths ---
        pbar = tqdm(
            zip(train_loader_old, train_loader_new),
            desc=f"Epoch {ep + 1}/{num_epochs}",
            unit="batch",
        )
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
    # TODO: Call LoRA Hub merge function here
    ViT_model = pl_model.model
    model, final_lora = lora_hub_learn(
        train_loader=hubloader, 
        lora_heads=lora_heads,
        LoRA_ViT_model=ViT_model,
        classifier=pl_model.classifier,                
        )
    pl_model.model = model
    merged_classifier = {}
    if hasattr(pl_model, "classifier") and isinstance(pl_model.classifier, nn.Linear):
        merged_classifier["weight"] = pl_model.classifier.weight.detach().cpu()
        if pl_model.classifier.bias is not None:
            merged_classifier["bias"] = pl_model.classifier.bias.detach().cpu()

    # 7. Clean up model by baking weights and restoring original state
    _cleanup_and_bake_weights(pl_model)

    return pl_model, final_lora, merged_classifier


def LoraSoupsMerge_from_scratch(
    pl_model: nn.Module,
    train_loader: Iterable,
    lora_heads: List[Dict[str, torch.Tensor]],
    classifier_heads: Optional[List[Dict[str, torch.Tensor]]] = None,
    mode: str = "learnable",
    num_epochs: int = 1,
    lr: float = 1e-4,
    current_task: int = 2,  # Add current_task parameter
) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Merges all task adapters from scratch to create a single unified adapter.

    This function takes the original, individual adapters for all tasks seen so far
    (1...T) and merges them together in a single step. The training process uses a
    single data loader containing a mix of samples from all tasks.

    - **LoRA Merging**: Learns coefficients to combine all original LoRA adapters provided.
    - **Classifier Merging**: Creates a new classifier by uniformly averaging the weights
      of all original classifier heads provided.

    Args:
        pl_model (nn.Module): The base model (e.g., a Pytorch Lightning module).
        train_loader (Iterable): A single DataLoader containing samples from all tasks.
        lora_heads (List[Dict[str, torch.Tensor]]): A list of T LoRA state dicts,
            one for each original task adapter from task 1 to T.
        classifier_heads (Optional[List[Dict[str, torch.Tensor]]]): A list of T classifier
            state dicts, one for each original task.
        mode (str): Merging mode. "learnable" for trainable alpha, "static" for uniform alpha.
        num_epochs (int): Number of epochs to train the merging coefficients.
        lr (float): Learning rate for the optimizer.
        current_task (int): The number of the current task (e.g., 2, 3, ...). Used for logging.

    Returns:
        Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            - The model with the newly merged weights baked in.
            - The state dict of the new merged LoRA adapter.
            - The state dict of the new merged classifier head.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)
    pl_model.train()

    # --- Add model structure printout, only for the first merge task ---
    if current_task == 2:
        log.info("--- Model Structure ---")
        log.info(pl_model)
        log.info("-----------------------")

    # 1. Freeze base model and collect trainable parameters
    for p in pl_model.parameters():
        p.requires_grad_(False)
    trainables: List[nn.Parameter] = []

    # 2. Gather zero-padded IDs from the LoRA keys:
    # e.g. layer_ids = ["000", "007", "013", ...]
    layer_ids = sorted(
        {k.split("_")[-1] for k in lora_heads[0] if k.startswith("w_a_")}
    )

    # Fallback for the case where the first head is a delta-only merge result.
    if not layer_ids and lora_heads:
        log.warning(
            "Initial layer ID discovery failed (no 'w_a_' keys). "
            "Falling back to robust discovery from all heads."
        )
        layer_ids = sorted(
            list(
                set(
                    k.split("_")[-1]
                    for h in lora_heads
                    for k in h
                    if k.startswith("w_a_") or k.startswith("delta_")
                )
            )
        )

    log.debug(f"Discovered layer IDs: {layer_ids}")
    ranks_are_consistent = _are_ranks_consistent(lora_heads)

    # 3. For each layer, compute deltas and attach parametrization
    soup_params: List[_LoRaSoupParam] = []
    patched_modules: List[str] = []
    # Only target q and v layers (since these have been trained)
    suffixes = ("q", "k", "v", "proj")

    # For each ID, convert to a “clean” integer string:
    for lid in layer_ids:
        deltas = _compute_deltas_for_layer(lid, lora_heads, device)
        if not deltas:
            log.warning(
                f"No deltas computed for layer ID {lid}, skipping patch for this layer."
            )
            continue

        parametrization = _LoRaSoupParam(deltas, mode=mode).to(device)
        found_module_for_lid = False
        clean_id = str(int(lid))  # "000" → "0", "007" → "7"

        # Look through every submodule:
        for name, mod in pl_model.named_modules():
            # Match only Linear layers *and* ensure the name contains the block ID
            # e.g. name = "model.blocks.7.attn.q" for ViT structure
            if (
                isinstance(mod, nn.Linear)
                and f".blocks.{clean_id}.attn." in name
                and name.rsplit(".", 1)[-1] in suffixes
            ):
                delta_shape = parametrization.deltas.shape[1:]
                if mod.weight.shape != delta_shape:
                    log.debug(
                        f"Skipping patch for '{name}': shape mismatch. "
                        f"Weight: {mod.weight.shape}, Delta: {delta_shape}"
                    )
                    continue

                # This is one of the LoRA-targeted linears!
                log.info(f"Patching module '{name}' for layer ID {lid}")
                parametrize.register_parametrization(
                    mod, "weight", parametrization, unsafe=False
                )
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
        weights, biases = [], []
        for head in classifier_heads:
            w, b = _split_weight_bias(head)
            weights.append(w)
            if b is not None:
                biases.append(b)

        # --- Logic to handle both continual and from-scratch merging ---
        num_heads = len(classifier_heads)

        if num_heads == 2:
            # CONTINUAL case: Weighted average of old (merged) and new.
            w_old = (current_task - 1) / current_task
            w_new = 1 / current_task
            log.info(
                f"Continual classifier merge: Applying weights old={w_old:.3f}, new={w_new:.3f}"
            )

            merged_weight = w_old * weights[0] + w_new * weights[1]
            merged_bias = None
            if biases:
                merged_bias = w_old * biases[0] + w_new * biases[1]
        else:
            # FROM-SCRATCH case: Uniform average of all heads.
            log.info(
                f"From-scratch classifier merge: Uniformly averaging {num_heads} heads."
            )

            merged_weight = torch.stack(weights).mean(dim=0)
            merged_bias = None
            if biases:
                merged_bias = torch.stack(biases).mean(dim=0)

        num_classes, hidden_dim = merged_weight.shape
        clf = nn.Linear(hidden_dim, num_classes, bias=merged_bias is not None).to(
            device
        )
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
    lora_params = sum(p.alpha.numel() for p in soup_params if hasattr(p, "alpha"))
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
        # --- Iterate over the unified data loader ---
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {ep + 1}/{num_epochs}",
            unit="batch",
        )
        for batch in pbar:
            x, y = batch
            x = x.to(device)
            y = y.to(device).float()

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
        merged_lora = _merge_lora_weights(
            layer_ids, soup_params, lora_heads, ranks_are_consistent
        )

    merged_classifier = {}
    if hasattr(pl_model, "classifier") and isinstance(pl_model.classifier, nn.Linear):
        merged_classifier["weight"] = pl_model.classifier.weight.detach().cpu()
        if pl_model.classifier.bias is not None:
            merged_classifier["bias"] = pl_model.classifier.bias.detach().cpu()

    # 7. Clean up model by baking weights and restoring original state
    _cleanup_and_bake_weights(pl_model)

    return pl_model, merged_lora, merged_classifier