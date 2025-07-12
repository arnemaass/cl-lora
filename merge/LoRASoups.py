import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from typing import Iterable, List, Dict, Tuple, Optional

class _LoRaSoupParam(nn.Module):
    """
    Parametrization that adds a weighted sum of precomputed LoRA deltas.

    self.deltas: Tensor of shape [H, out_feats, in_feats]
    self.alpha:  Tensor of shape [H] (trainable or fixed)

    forward(W): returns W + sum_h alpha[h] * deltas[h]
    """
    def __init__(self,
                 deltas: List[torch.Tensor],  # each [out, in] = B @ A
                 mode: str = "learnable"):
        super().__init__()
        # stack H heads → [H, out, in]
        self.register_buffer('deltas', torch.stack(deltas))
        H = self.deltas.size(0)
        if mode == "learnable":
            # initialize alpha_h = 1/H
            init = 1.0 / H
            self.alpha = nn.Parameter(torch.full((H,), init))
        else:
            # static alpha_h = 1/H
            self.register_buffer('alpha', torch.full((H,), 1.0 / H))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # delta = sum_h alpha_h · deltas[h]
        delta = torch.einsum('h,hod->od', self.alpha, self.deltas)
        return W + delta
    
def split_weight_bias(head: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # pull out the 2D weight and optional 1D bias
    weight = next(v for v in head.values() if v.ndim == 2)
    bias   = next((v for v in head.values() if v.ndim == 1), None)
    return weight, bias


def LoraSoupsMerge(
    pl_model:           nn.Module,
    train_loader_old:       Iterable,
    train_loader_new:       Iterable,
    lora_heads:         List[Dict[str, torch.Tensor]],   # each has w_a_{lid}, w_b_{lid}
    classifier_heads:   Optional[List[Dict[str, torch.Tensor]]] = None,
    mode:               str   = "learnable",  # "learnable" or "static"
    num_epochs:         int   = 1,
    lr:                  float = 1e-4,
) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Merge multiple LoRA adapters via per-head, per-layer α coefficients.

    1. Freeze all base-model parameters.
    2. Compute each layer's LoRA deltas: delta_i^l = B_i^l @ A_i^l.
    3. Attach a _LoRaSoupParam to each target nn.Linear weight:
         W' = W + sum_i alpha_i^l delta_i^l
    4. If mode="learnable": alpha’s are trainable; else alpha_i^l = 1/n.
    5. Unfreeze (or create) the classification head.
    6. Train only the alpha’s (and classifier) via gradient descent.
    7. Return (merged_model, merged_lora_weights, merged_classifier_weights).

    The update per layer l is computed as:
        delta_W^l = sum_i alpha_i^l · (B_i^l @ A_i^l)  
    Raw LaTeX: $$\Delta W^l = \sum_i \alpha_i^l\,B_i^l A_i^{l\,T}$$
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    # 1. freeze base model
    for p in pl_model.parameters():
        p.requires_grad_(False)

    # collect layer IDs from keys like "w_a_000"
    layer_ids = sorted({k.split('_')[-1] for k in lora_heads[0] if k.startswith("w_a_")})

    soup_params: List[_LoRaSoupParam] = []
    trainables: List[nn.Parameter] = []

    # 2. for each layer, compute deltas and attach parametrization
    for lid in layer_ids:
        deltas = []
        for hd in lora_heads:
            A = hd[f"w_a_{lid}"].to(device)   # [r, in]
            B = hd[f"w_b_{lid}"].to(device)   # [out, r]
            deltas.append(B @ A)              # → [out, in]

        zp = _LoRaSoupParam(deltas, mode=mode).to(device)
        soup_params.append(zp)
        clean = str(int(lid))  # "000"→"0"

        # patch only linears whose shape matches delta_W^l:
        # $$\Delta W^l = \sum_i \alpha_i^l\,B_i^l A_i^{l\,T}$$
        for name, mod in pl_model.named_modules():
            if isinstance(mod, nn.Linear) and f".{clean}." in name:
                # shape check prevents size mismatch
                delta = torch.einsum('h,hod->od', zp.alpha if isinstance(zp.alpha, torch.Tensor) else zp.alpha,
                                     zp.deltas)
                if mod.weight.shape == delta.shape:
                    parametrize.register_parametrization(mod, "weight", zp, unsafe=False)

        # collect alpha params if learnable
        if mode == "learnable":
            trainables.append(zp.alpha)

    # 3. classifier head (robust extraction)
    if classifier_heads:
        head = classifier_heads[-1]
        w, b = split_weight_bias(head)
        w, b = w.to(device), b.to(device) if b is not None else None

        clf = nn.Linear(w.size(1), w.size(0), bias=(b is not None)).to(device)
        with torch.no_grad():
            clf.weight.copy_(w)
            if b is not None:
                clf.bias.copy_(b)
        pl_model.classifier = clf
        for p in clf.parameters():
            p.requires_grad_(True)
        trainables += list(pl_model.classifier.parameters())

        # Override forward to apply the classifier head.
        if not hasattr(pl_model, "_orig_forward"):
            pl_model._orig_forward = pl_model.forward
            def _forward_with_classifier(x):
                feats = pl_model._orig_forward(x)
                # if features have the expected dimension, pass through classifier
                if feats.ndim == 2 and feats.shape[1] == clf.weight.shape[1]:
                    return pl_model.classifier(feats)
                return feats
            pl_model.forward = _forward_with_classifier

    # 4. optimizer & loss
    optim   = torch.optim.Adam(trainables, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # multilabel float targets

 # ───────────────────────────────────────────────────────────────
    # 5. optimizer & multi-task training loop (old + new data)
    # ───────────────────────────────────────────────────────────────

    def _run(loader_old, loader_new):
        for (x_old, y_old), (x_new, y_new) in zip(loader_old, loader_new):
            # move to device
            x_old, x_new = x_old.to(device), x_new.to(device)
            # keep floats [B, C] for BCE
            y_old, y_new = y_old.to(device).float(), y_new.to(device).float()

            # forward
            out_old = pl_model(x_old)  # [B, C]
            out_new = pl_model(x_new)  # [B, C]

            # compute losses
            loss_old = loss_fn(out_old, y_old)
            loss_new = loss_fn(out_new, y_new)

            # total multi-task objective:
            loss = loss_old + loss_new 

            optim.zero_grad()
            loss.backward()
            optim.step()

    # run for num_epochs
    for ep in range(num_epochs):
        _run(train_loader_old, train_loader_new)
        print(f"[LoraSoupsMerge] epoch {ep+1}/{num_epochs}  "
              f"layers={len(layer_ids)}  heads={len(lora_heads)}")

    # 6. Build merged_lora: weighted combination of original LoRA heads
    merged_lora: Dict[str, torch.Tensor] = {}
    for lid, zp in zip(layer_ids, soup_params):
        with torch.no_grad():
            # Get the learned/fixed alpha weights
            alpha_weights = zp.alpha.detach().cpu()  # [H]
            
            # Collect original A and B matrices for this layer
            A_matrices = []  # Will be [H, r, d]
            B_matrices = []  # Will be [H, o, r]
            
            for hd in lora_heads:
                A = hd[f"w_a_{lid}"]  # [r, d]
                B = hd[f"w_b_{lid}"]  # [o, r]
                A_matrices.append(A)
                B_matrices.append(B)
            
            # Check if all ranks are the same
            ranks = [A.shape[0] for A in A_matrices]
            if len(set(ranks)) > 1:
                # Different ranks - use delta-based merging instead
                deltas = []
                for A, B in zip(A_matrices, B_matrices):
                    delta = B @ A  # [o, d]
                    deltas.append(delta)
                
                # Stack deltas and apply weights
                delta_stacked = torch.stack(deltas)  # [H, o, d]
                merged_delta = torch.einsum('h,hod->od', alpha_weights, delta_stacked)
                
                # For different ranks, we can't create a single LoRA representation
                # Store as a single delta weight instead
                merged_lora[f"delta_{lid}"] = merged_delta.cpu()
            else:
                # Same ranks - use original Zip-LoRA approach
                # Stack into tensors
                A_stacked = torch.stack(A_matrices)  # [H, r, d]
                B_stacked = torch.stack(B_matrices)  # [H, o, r]
                
                # Get dimensions
                H, r, d = A_stacked.shape
                o = B_stacked.shape[1]
                
                # Scale A matrices by alpha weights
                A_scaled = A_stacked * alpha_weights.unsqueeze(1).unsqueeze(2)  # [H, r, d]
                
                # Compute reference delta for verification
                delta_ref = torch.einsum("hor,hrd->hod", B_stacked, A_scaled).sum(0)
                
                # Flatten into one big LoRA (Zip-LoRA style)
                A_tot = A_scaled.reshape(H * r, d) # [H·r, d]
                B_tot = B_stacked.permute(1, 0, 2).reshape(o, H * r) # [o, H·r]
                
                # Reconstruct delta for sanity check
                delta_recon = B_tot @ A_tot # [o, d]
                
                # Sanity check
                if not torch.allclose(delta_recon, delta_ref, atol=1e-6):
                    raise RuntimeError(f"reconstruction mismatch in layer {lid}")
                
                # Store merged parameters
                merged_lora[f"w_a_{lid}"] = A_tot.cpu()
                merged_lora[f"w_b_{lid}"] = B_tot.cpu()

    # 7. Clean up parametrizations before returning (to avoid serialization issues)
    for name, mod in list(pl_model.named_modules()):
        if isinstance(mod, nn.Linear) and hasattr(mod, "parametrizations"):
            # Remove parametrizations to make model serializable
            if hasattr(mod.parametrizations, "weight"):
                # Get the current parametrized weight
                current_weight = mod.weight.detach()
                # Remove the parametrization
                parametrize.remove_parametrizations(mod, "weight")
                # Optionally bake the learned weights back into the base model
                # mod.weight.data = current_weight

    # 8. Handle forward function for serialization
    # If classifier was added, we need to keep the modified forward but make it serializable
    if hasattr(pl_model, "_orig_forward") and hasattr(pl_model, "classifier"):
        # Create a proper method instead of a closure to make it serializable
        def forward_with_classifier(self, x):
            feats = self._orig_forward(x)
            # if features have the expected dimension, pass through classifier
            if feats.ndim == 2 and feats.shape[1] == self.classifier.weight.shape[1]:
                return self.classifier(feats)
            return feats
        
        # Bind the method to the model
        import types
        pl_model.forward = types.MethodType(forward_with_classifier, pl_model)
        
    elif hasattr(pl_model, "_orig_forward"):
        # Only restore original forward if no classifier was added
        pl_model.forward = pl_model._orig_forward
        delattr(pl_model, "_orig_forward")

    # 9. build merged_classifier dict (alternative approach)
    merged_classifier = {}
    classifier_mod = getattr(pl_model, "classifier", None)
    if isinstance(classifier_mod, nn.Linear):
        W = classifier_mod.weight.detach().cpu()
        out_feats, in_feats = W.shape
        key = f"fc_{in_feats}in_{out_feats}out"
        merged_classifier[key] = W
        
    return pl_model, merged_lora, merged_classifier
