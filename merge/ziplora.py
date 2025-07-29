import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from typing import List, Dict, Iterable, Optional

# ───────────────────────────────────────────────────────────────────
# 0.  Zip-LoRA parametrisation   (2 heads, per-column merger vectors)
# ───────────────────────────────────────────────────────────────────
class _ZipParam(nn.Module):
    """
    A parametrization module that merges two LoRA heads into a single delta matrix
    with learnable per-column scaling vectors.

    Args:
        A_list (List[torch.Tensor]): List of A matrices for each LoRA head, each of shape [r_i, d],
                                     where r_i is the rank for head i.
        B_list (List[torch.Tensor]): List of B matrices for each LoRA head, each of shape [o, r_i],
                                     where o is the output dimension.
        init (float): Initial value for the learnable merger vector `m`.

    Process:
        - Pads each A and B matrix so that all ranks r_i match the maximum rank r_max across heads.
        - Stacks the padded matrices into tensors of shapes [H, r_max, d] (for A) and [H, o, r_max] (for B),
          where H is the number of heads (usually 2).
        - Introduces a learnable merger vector `m` for each head (shape [H, d]) to scale A.
    """

    def __init__(self, A_list, B_list, init=0.5):
        super().__init__()

        # Find the maximum rank across all heads
        r_list = [A.shape[0] for A in A_list]
        r_max = max(r_list)
        A_padded = []
        B_padded = []

        # Pad A and B for each head so they all have the same rank (r_max)
        for A, B in zip(A_list, B_list):
            r_i, d = A.shape      # A is [r_i, d]
            o, _  = B.shape       # B is [o, r_i]
            if r_i < r_max:
                pad = r_max - r_i
                A = F.pad(A, (0, 0, 0, pad))  # Pad rows up to r_max → [r_max, d]
                B = F.pad(B, (0, pad, 0, 0))  # Pad columns up to r_max → [o, r_max]
            A_padded.append(A)
            B_padded.append(B)

        # Store padded matrices as buffers (not learnable parameters)
        self.register_buffer("A", torch.stack(A_padded))  # [H, r_max, d]
        self.register_buffer("B", torch.stack(B_padded))  # [H, o, r_max]

        H, _, d = self.A.shape
        # One learnable merger vector per head (shape [H, d])
        self.m = nn.Parameter(torch.full((H, d), init))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """
        Applies the Zip-LoRA delta to a given weight matrix.

        Args:
            W (torch.Tensor): The original weight matrix of the layer.

        Returns:
            torch.Tensor: The modified weight matrix (W + delta).
        """
        # Scale A by the merger vector (broadcast along rank dimension)
        A_scaled = self.A * self.m.unsqueeze(1)  # [H, r_max, d]

        # Compute the per-head delta: B[h] @ A_scaled[h] → [H, o, d]
        delta = torch.einsum("hor,hrd->hod", self.B, A_scaled).sum(0)  # Sum across heads

        return W + delta

# ───────────────────────────────────────────────────────────────────
# 1.  Main API
# ───────────────────────────────────────────────────────────────────
def ZipLoRaMerge(
        pl_model:          nn.Module,
        train_loader_old:  Iterable,
        train_loader_new:  Iterable,
        lora_heads:        List[Dict[str, torch.Tensor]],   # exactly 2 LoRA heads
        classifier_heads:  Optional[List[Dict[str, torch.Tensor]]] = None,
        step:              int   = 0,
        lr:                float = 1e-4,
        num_epochs:        int   = 3,
        d_old: float = 1.0,
        d_new: float = 1.0,
        d_cos: float = 0.1,  # strength of the similarity penalty
):
    """
    Merges two LoRA-modified models into one model by learning per-column merger
    vectors (`m`) for each LoRA layer.

    The process:
      1. Attaches a `_ZipParam` parametrization to each matching linear layer.
      2. Trains only the merger vectors (and optionally a classifier) using data
         from both old and new tasks.
      3. Outputs the merged LoRA weights and (if available) the merged classifier.

    Args:
        pl_model (nn.Module): The PyTorch model to patch with Zip-LoRA layers.
        train_loader_old (Iterable): Dataloader for old task data.
        train_loader_new (Iterable): Dataloader for new task data.
        lora_heads (List[Dict[str, torch.Tensor]]): Two LoRA state dicts (old & new).
        classifier_heads (Optional[List[Dict[str, torch.Tensor]]]): Classifier weights for old & new tasks.
        step (int): Continual learning step index (affects classifier weighting).
        lr (float): Learning rate for the merger parameters.
        num_epochs (int): Number of epochs to train the merger.
        d_old, d_new (float): Loss weights for old/new tasks.
        d_cos (float): Weight for the cosine similarity penalty between heads.

    Returns:
        tuple: (merged_model, merged_lora, merged_classifier)
            merged_model (nn.Module): Model with trained Zip-LoRA.
            merged_lora (dict): Dict of merged LoRA weights (A and B matrices).
            merged_classifier (dict): Dict of merged classifier weights (if any).
    """


    assert len(lora_heads) == 2, "ZipLoRaMerge currently requires exactly two LoRA heads"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    # Remove any pre-existing parametrizations (clean slate)
    for name, mod in pl_model.named_modules():
        if isinstance(mod, nn.Linear) and parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)

    # Freeze all parameters except those we explicitly set as trainable
    for p in pl_model.parameters():
        p.requires_grad_(False)

    # Extract unique layer IDs from LoRA keys (e.g., "w_a_000" → "0")
    layer_ids = sorted({k.split('_')[-1] for k in lora_heads[0] if k.startswith("w_a_")})

    trainables, patched = [], []
    zip_params: List[_ZipParam] = []  # Stores the parametrizations for cosine penalty

    # Attach Zip-LoRA to each relevant linear layer
    for lid in layer_ids:
        clean_id = str(int(lid))  # "000" → "0", "007" → "7"
        A_list = [hd[f"w_a_{lid}"].to(device) for hd in lora_heads]
        B_list = [hd[f"w_b_{lid}"].to(device) for hd in lora_heads]
        zp = _ZipParam(A_list, B_list).to(device)

        added_to_list = False
        suffixes = ("q", "k", "v", "proj")  # Only patch specific attention components
        for name, mod in pl_model.named_modules():
            if isinstance(mod, nn.Linear) and f".{clean_id}." in name:
                if name.rsplit('.', 1)[-1] in suffixes:
                    parametrize.register_parametrization(mod, "weight", zp, unsafe=False)
                    if not added_to_list:
                        zip_params.append(zp)
                        added_to_list = True
                    patched.append(name)
        if patched:  # Add the merger vector `m` to trainables
            trainables.append(zp.m)

    if not patched:
        raise RuntimeError(
            "ZipLoRaMerge: No matching Linear layers found.\n"
            "Check module names; example first Linear name is "
            f"'{next(n for n,_ in pl_model.named_modules() if isinstance(_, nn.Linear))}'"
        )

    # ----------------------------------------------------------------------
    #  CLASSIFIER MERGING
    # ----------------------------------------------------------------------
    if classifier_heads and len(classifier_heads) == 2:
        # Weight old/new classifiers based on step
        w_old, w_new = (0.0, 1.0) if step <= 0 else ((step - 1) / step, 1.0 / step)

        def split_weight_bias(head):
            """Helper to extract weight and bias from a classifier head."""
            weight = next(v for v in head.values() if v.ndim == 2)
            bias = next((v for v in head.values() if v.ndim == 1), None)
            return weight, bias

        weight_old, bias_old = split_weight_bias(classifier_heads[0])
        weight_new, bias_new = split_weight_bias(classifier_heads[1])

        merged_weight = w_old * weight_old + w_new * weight_new
        merged_bias = None
        if bias_old is not None and bias_new is not None:
            merged_bias = w_old * bias_old + w_new * bias_new

        num_classes, hidden_dim = merged_weight.shape
        classifier = nn.Linear(hidden_dim, num_classes, bias=merged_bias is not None).to(device)
        with torch.no_grad():
            classifier.weight.copy_(merged_weight)
            if merged_bias is not None:
                classifier.bias.copy_(merged_bias)

        pl_model.classifier = classifier
        for p in classifier.parameters():
            p.requires_grad_(True)
        trainables += list(classifier.parameters())

        print(f"[ZipLoRaMerge] created classifier   "
              f"(old w={w_old:.3f}, new w={w_new:.3f})   "
              f"shape={classifier.weight.shape}")
    else:
        print("[ZipLoRaMerge] no classifier heads provided – skipping merge")

    # Patch model forward to include classifier
    if not hasattr(pl_model, "_orig_forward"):
        pl_model._orig_forward = pl_model.forward
        def _forward_with_head(x):
            feats = pl_model._orig_forward(x)
            if feats.ndim == 2 and feats.shape[1] == hidden_dim:
                feats = pl_model.classifier(feats)
            return feats
        pl_model.forward = _forward_with_head

    # Report trainable parameters
    lora_params = sum(p.numel() for zp in zip_params for p in (zp.m,))
    cls_params = sum(p.numel() for p in classifier.parameters()) if 'classifier' in locals() else 0
    print(f"[ZipLoRaMerge]  trainable LoRA-merge params : {lora_params:,}")
    print(f"[ZipLoRaMerge]  trainable classifier params : {cls_params:,}")
    print(f"[ZipLoRaMerge]  total trainables            : {lora_params + cls_params:,}")

    # Optimizer and loss
    optim = torch.optim.Adam(trainables, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Inner training loop for old & new tasks
    def _run(loader_old, loader_new):
        for (x_old, y_old), (x_new, y_new) in zip(loader_old, loader_new):
            x_old, x_new = x_old.to(device), x_new.to(device)
            y_old, y_new = y_old.to(device).float(), y_new.to(device).float()

            out_old = pl_model(x_old)
            out_new = pl_model(x_new)

            loss_old = loss_fn(out_old, y_old)
            loss_new = loss_fn(out_new, y_new)

            # Cosine penalty between merger vectors (encourages similarity)
            cos_pen = sum(
                F.cosine_similarity(zp.m[0], zp.m[1], dim=0).abs()
                for zp in zip_params
            ) / len(zip_params)

            # Weighted sum of losses
            term_old = d_old * loss_old
            term_new = d_new * loss_new
            term_cos = d_cos * cos_pen

            loss = term_old + term_new + term_cos

            optim.zero_grad()
            loss.backward()
            optim.step()

    # Train merger parameters
    for ep in range(num_epochs):
        print(f"Epoch {ep + 1}/{num_epochs}")
        _run(train_loader_old, train_loader_new)
        print(f"[ZipLoRaMerge] epoch {ep + 1}/{num_epochs}  patched={len(patched)} linears")

    # ------------------------------------------------------------------
    #  Build merged LoRA weights
    # ------------------------------------------------------------------
    merged_lora = {}
    for lid, zp in zip(layer_ids, zip_params):
        with torch.no_grad():
            H, r, d = zp.A.shape
            o = zp.B.shape[1]

            A_scaled = zp.A * zp.m.unsqueeze(1)
            delta_ref = torch.einsum("hor,hrd->hod", zp.B, A_scaled).sum(0)

            A_tot = A_scaled.reshape(H * r, d)  # Merge heads
            B_tot = zp.B.permute(1, 0, 2).reshape(o, H * r)

            # Sanity check: recomputed delta should match reference
            delta_recon = B_tot @ A_tot
            if not torch.allclose(delta_recon, delta_ref, atol=1e-6):
                raise RuntimeError(f"reconstruction mismatch in layer {lid}")

        merged_lora[f"w_a_{lid}"] = A_tot.cpu()
        merged_lora[f"w_b_{lid}"] = B_tot.cpu()

    # ------------------------------------------------------------------
    #  Build merged classifier dict
    # ------------------------------------------------------------------
    classifier_mod = getattr(pl_model, "classifier", None)
    merged_classifier = {}
    if isinstance(classifier_mod, nn.Linear):
        W = classifier_mod.weight.detach().cpu()
        out_feats, in_feats = W.shape
        key = f"fc_{in_feats}in_{out_feats}out"
        merged_classifier[key] = W

    return pl_model, merged_lora, merged_classifier