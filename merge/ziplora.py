import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List

"""
def ZipLoRaMerge(pl_model,train_loader_old,train_loader_new,lora_heads,classifier_heads,step,lr,num_epochs=100,mode="learnable"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    # 1. Freeze all base model parameters
    for param in pl_model.parameters():
        param.requires_grad = False

    # 2. Extract LoRA layers and organize by layer index
    num_heads = len(lora_heads)
    print("num heads:" +str(num_heads))

    # Find all layer indices from the first LoRA head
    layer_indices = set()
    for key in lora_heads[0].keys():
        if key.startswith('w_a_') or key.startswith('w_b_'):
            layer_idx = key.split('_')[-1]  # Extract XXX from w_a_XXX or w_b_XXX
            layer_indices.add(layer_idx)

    layer_indices = sorted(list(layer_indices))
    print("layer indices:"+ str(layer_indices))
    num_layers = len(layer_indices)
    print("num layers:" +str(num_layers))

    merger_vecs = nn.ParameterDict()

    for h, head in enumerate(lora_heads):  # iterate over *all* heads
        for layer_idx in layer_indices:  # … and over *all* layers
            in_features = head[f"w_a_{layer_idx}"].shape[1]
            key = f"layer_{layer_idx}_head_{h}"  # unique key per (layer, head)

            #  initialise with equal weighting (½, ½) when you have exactly 2 heads
            init = torch.full((in_features,), 1.0 / len(lora_heads), device=device)

            merger_vecs[key] = nn.Parameter(init, requires_grad=True)

    print(merger_vecs)

    # add downstream head naively

    # loss one: remain fucntionality of old model in merged model (calc loss on old data in merged model)

    # loss two: remainf functionality of new model in merged mode (can this be done with the right proportion of data loss 1)

    # loss three: cossim of merging vectors
    #for i in range(num_epochs):
        # gradient step: how do a step whe we only change merger coeffitions on loss 1 and 2 and then add cossim loss



    #lora_merge = lora_old # can we return w or does it need to be a and b
    #return merged_model, merged_lora, merged_classifier
    return None


# ───────────────────────────────────────────────────────────────────────────────
# 1.  Zip-style linear layer  (slightly adapted)
# ───────────────────────────────────────────────────────────────────────────────
class ZipLoRALinearLayer(nn.Module):
    """
# Holds two frozen LoRA-updates and two learnable merger vectors.
    #`get_ziplora_weight()` returns  B1 @ (A1 ⊙ m1) + B2 @ (A2 ⊙ m2)
"""
    def __init__(self,
                 weight_1: torch.Tensor,      #  [out, in]  = B1 @ A1
                 weight_2: torch.Tensor,      #  [out, in]  = B2 @ A2
                 init_val_1: float = 0.5,
                 init_val_2: float = 0.5):
        super().__init__()
        out, in_feat = weight_1.shape
        # frozen low-rank updates
        self.register_buffer("weight_1", weight_1)
        self.register_buffer("weight_2", weight_2)
        # learnable per-column scalars
        self.merger_1 = nn.Parameter(torch.full((in_feat,), init_val_1))
        self.merger_2 = nn.Parameter(torch.full((in_feat,), init_val_2))

    # utility
    def get_ziplora_weight(self) -> torch.Tensor:
        return self.weight_1 * self.merger_1 + self.weight_2 * self.merger_2

# helper so we can call .set_lora_layer() on any nn.Linear
class LoRAAwareLinear(nn.Linear):
    def set_lora_layer(self, lora_layer: ZipLoRALinearLayer):
        self.ziplora = lora_layer               # attach once

    def forward(self, x):
        if hasattr(self, "ziplora"):
            W = self.weight + self.ziplora.get_ziplora_weight()
            return F.linear(x, W, self.bias)
        else:
            return super().forward(x)
"""
# zip_merge.py
import torch, torch.nn as nn
from torch.nn.utils import parametrize
from typing import List, Dict, Iterable, Optional

# ───────────────────────────────────────────────────────────────────
# 0.  Zip-LoRA parametrisation   (2 heads, per-column merger vectors)
# ───────────────────────────────────────────────────────────────────
class _ZipParam(nn.Module):
    def __init__(self, A_list, B_list, init=0.5):
        """
        A_list: list of [r_i × d] tensors
        B_list: list of [o   × r_i] tensors
        We zero-pad each (A_i,B_i) so all r_i → r_max, then stack.
        """
        super().__init__()

        # find the maximum rank across heads
        r_list = [A.shape[0] for A in A_list]
        r_max = max(r_list)
        A_padded = []
        B_padded = []

        # pad each head up to r_max
        for A, B in zip(A_list, B_list):
            r_i, d = A.shape      # A is [r_i, d]
            o, _  = B.shape       # B is [o, r_i]
            if r_i < r_max:
                pad   = r_max - r_i
                # pad A on dim0 (rows) up to r_max
                A = F.pad(A, (0, 0, 0, pad))     # → [r_max, d]
                # pad B on dim1 (cols) up to r_max
                B = F.pad(B, (0, pad, 0, 0))     # → [o, r_max]
            A_padded.append(A)
            B_padded.append(B)

        # now all A_padded[i] have shape [r_max, d], andB_padded[i] [o, r_max]
        self.register_buffer("A", torch.stack(A_padded))  # [H, r_max, d]
        self.register_buffer("B", torch.stack(B_padded))  # [H, o, r_max]

        H, _, d = self.A.shape
        # one learnable merger vector per head, length d
        self.m = nn.Parameter(torch.full((H, d), init))

    def forward(self, W):
        # scale and merge:
        #   A_scaled[h] = A[h] ⊙ m[h]  → [H, r_max, d]
        A_scaled = self.A * self.m.unsqueeze(1)
        # per-head delta:  B[h] @ A_scaled[h]  → [H, o, d]
        # sum heads → delta [o, d]
        delta = torch.einsum("hor,hrd->hod", self.B, A_scaled).sum(0)
        return W + delta
# ───────────────────────────────────────────────────────────────────
# 1.  main API
# ───────────────────────────────────────────────────────────────────
def ZipLoRaMerge(
        pl_model:          nn.Module,
        train_loader_old:  Iterable,
        train_loader_new:  Iterable,
        lora_heads:        List[Dict[str, torch.Tensor]],   # exactly 2
        classifier_heads:  Optional[List[Dict[str, torch.Tensor]]] = None,
        step:              int   = 0,       # kept for API compat
        lr:                float = 1e-4,
        num_epochs:        int   = 3,
        d_old: float = 1.0,
        d_new: float = 1.0,
        d_cos: float = 0.1,         #  ← strength of the similarity penalty
):

    """
    * Finds every nn.Linear whose name contains '.<clean_id>.' where
      <clean_id> is the integer value of the zero-padded layer id in the
      LoRA keys (w_a_000  →  clean_id '0').
    * Attaches a Zip-LoRA parametrisation (two heads) to that Linear
      and, if available, to sibling linears 'k', 'v', 'proj' in the same
      attention block.
    * Optimises only the merger vectors  (and the classifier if provided).
    * Returns the trained model.
    """
    print(lora_heads[0])
    print(classifier_heads[0])
    #print(pl_model)

    assert len(lora_heads) == 2, "need exactly two LoRA heads"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    # remove old parameterization
    for name, mod in pl_model.named_modules():
        if isinstance(mod, nn.Linear):
            # this will strip off _any_ existing parametrizations on weight
            if isinstance(mod, nn.Linear) and parametrize.is_parametrized(mod, "weight"):
                parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)

    print(pl_model)

    # freeze everything
    for p in pl_model.parameters():
        p.requires_grad_(False)

    # gather zero-padded ids from LoRA keys
    layer_ids = sorted({k.split('_')[-1] for k in lora_heads[0]
                        if k.startswith("w_a_")})

    trainables, patched = [], []
    zip_params: List[_ZipParam] = []  # <─  ADD THIS ONCE near the top of ZipLoRaMerge

    for lid in layer_ids:
        clean_id = str(int(lid))            # '000'→'0', '007'→'7'
        A_list = [hd[f"w_a_{lid}"].to(device) for hd in lora_heads]
        B_list = [hd[f"w_b_{lid}"].to(device) for hd in lora_heads]
        zp = _ZipParam(A_list, B_list).to(device)

        added_to_list = False  # ①  MOVE this line **outside** inner loop

        # patch q / k / v / proj if they exist in this block
        suffixes = ("q", "k", "v", "proj")
        for name, mod in pl_model.named_modules():
            if isinstance(mod, nn.Linear) and f".{clean_id}." in name:
                if name.rsplit('.', 1)[-1] in suffixes:
                    parametrize.register_parametrization(mod, "weight", zp, unsafe=False)
                    if not added_to_list:  # ADD THESE TWO LINES
                        zip_params.append(zp)  # keep handle for cosine term
                        added_to_list = True
                    patched.append(name)
        if patched:                         # only add trainables once per block
            trainables.append(zp.m)

    print(len(trainables))
    print(len(patched))
    if not patched:
        raise RuntimeError("ZipLoRaMerge: no matching Linear layers found.\n"
                           "Check module names; example first Linear name "
                           f"is '{next(n for n,_ in pl_model.named_modules() if isinstance(_, nn.Linear))}'")

    # ----------------------------------------------------------------------
    #  OPTIONAL CLASSIFIER MERGING  (robust to arbitrary key names)
    # ----------------------------------------------------------------------
    if classifier_heads and len(classifier_heads) == 2:
        if step <= 0:
            w_old, w_new = 0.0, 1.0  # bootstrap (no old tasks yet)
        else:
            w_old = (step - 1) / step
            w_new = 1.0 / step

        # -------- helper: split weight / bias by tensor rank ----------------
        def split_weight_bias(head):
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
        classifier = nn.Linear(hidden_dim, num_classes,
                               bias=merged_bias is not None).to(device)
        with torch.no_grad():
            classifier.weight.copy_(merged_weight)
            if merged_bias is not None:
                classifier.bias.copy_(merged_bias)

        # expose it (pick any attribute name you like)
        pl_model.classifier = classifier
        for p in classifier.parameters():
            p.requires_grad_(True)
        trainables += list(classifier.parameters())

        print(f"[ZipLoRaMerge] created classifier   "
              f"(old w={w_old:.3f}, new w={w_new:.3f})   "
              f"shape={classifier.weight.shape}")
    else:
        print("[ZipLoRaMerge] no classifier heads provided – skipping merge")

    if not hasattr(pl_model, "_orig_forward"):
        pl_model._orig_forward = pl_model.forward  # backup

        def _forward_with_head(x):
            feats = pl_model._orig_forward(x)
            # if the last dim equals hidden_dim we assume it's features
            if feats.ndim == 2 and feats.shape[1] == hidden_dim:
                feats = pl_model.classifier(feats)
            return feats

        pl_model.forward = _forward_with_head

    lora_params = sum(p.numel() for zp in zip_params for p in (zp.m,))
    cls_params = sum(p.numel() for p in classifier.parameters()) if 'classifier' in locals() else 0
    print(f"[ZipLoRaMerge]  trainable LoRA-merge params : {lora_params:,}")
    print(f"[ZipLoRaMerge]  trainable classifier params : {cls_params:,}")
    print(f"[ZipLoRaMerge]  total trainables            : {lora_params + cls_params:,}")

    # optimiser
    optim = torch.optim.Adam(trainables, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # and keep yb as float (0/1)

    def _run(loader_old, loader_new):
        for (x_old, y_old), (x_new, y_new) in zip(loader_old, loader_new):
            # move to device ───────────────────────────────────────────
            x_old, x_new = x_old.to(device), x_new.to(device)
            y_old, y_new = y_old.to(device).float(), y_new.to(device).float()
            #  ^^^^^^^^^^  keep shape [B, 19] — do *not* squeeze / one-hot

            # forward ──────────────────────────────────────────────────
            out_old = pl_model(x_old)  # [B, 19]
            out_new = pl_model(x_new)  # [B, 19]

            # losses ───────────────────────────────────────────────────
            loss_old = loss_fn(out_old, y_old)  # BCEWithLogitsLoss
            loss_new = loss_fn(out_new, y_new)

            # cosine penalty (unchanged) ───────────────────────────────
            cos_pen = 0.0
            for zp in zip_params:
                cos_pen += F.cosine_similarity(zp.m[0], zp.m[1], dim=0).abs()
            cos_pen /= len(zip_params)

            loss = d_old * loss_old + d_new * loss_new + d_cos * cos_pen

            optim.zero_grad()
            loss.backward()
            optim.step()
    for ep in range(num_epochs):
        _run(train_loader_old,train_loader_new)
        print(f"[ZipLoRaMerge] epoch {ep+1}/{num_epochs}  patched={len(patched)} linears")

    # ------------------------------------------------------------------
    #  Build merged_lora: one (A_tot, B_tot) + dense delta per layer
    # ------------------------------------------------------------------
    merged_lora = {}
    for lid, zp in zip(layer_ids, zip_params):
        with torch.no_grad():
            # shapes
            H, r, d = zp.A.shape  # [H, r, d]
            o = zp.B.shape[1]  # zp.B is [H, o, r]

            # 1) original scaled heads → Δ_ref ∈ ℝ^{o×d}
            A_scaled = zp.A * zp.m.unsqueeze(1)  # [H, r, d]
            delta_ref = torch.einsum("hor,hrd->hod", zp.B, A_scaled).sum(0)

            # 2) flatten into one big LoRA
            A_tot = A_scaled.reshape(H * r, d)  # [H·r, d]
            B_tot = zp.B.permute(1, 0, 2).reshape(o, H * r)  # [o, H·r]

            # 3) reconstruct
            delta_recon = B_tot @ A_tot  # [o, d]

        # 4) sanity check
        if not torch.allclose(delta_recon, delta_ref, atol=1e-6):
            raise RuntimeError(f"reconstruction mismatch in layer {lid}")

        # store everything
        #merged_lora[f"delta_{lid}"] = delta.cpu()
        merged_lora[f"w_a_{lid}"]    = A_tot.cpu()
        merged_lora[f"w_b_{lid}"]    = B_tot.cpu()

    # ------------------------------------------------------------------
    #  Return the merged model + all the LoRA outputs
    # ------------------------------------------------------------------
    # 1) Grab the classifier module (if it exists)
    classifier_mod = getattr(pl_model, "classifier", None)

    # 2) Build merged_classifier as a dict
    merged_classifier = {}
    if isinstance(classifier_mod, nn.Linear):
        # extract weight [out_feats, in_feats]
        W = classifier_mod.weight.detach().cpu()
        out_feats, in_feats = W.shape

        # build key 'fc_{in}in_{out}out'
        key = f"fc_{in_feats}in_{out_feats}out"

        # insert into dict
        merged_classifier[key] = W
    return pl_model, merged_lora, merged_classifier

"""
    merged_lora = {}
    for lid, zp in zip(layer_ids, zip_params):
        # compute ΔW = B @ (A ⊙ m)   for this layer
        with torch.no_grad():
            delta = torch.einsum("hor,rd->hod", zp.B, zp.A[0] * zp.m[0]) \
                    + torch.einsum("hor,rd->hod", zp.B, zp.A[1] * zp.m[1])
        merged_lora[f"delta_{lid}"] = delta.cpu()

    # 2️⃣  Keep a handle to the (possibly newly-created) classifier
    merged_classifier = getattr(pl_model, "classifier", None)

    # 3️⃣  Return all three objects in the same order LoRASoupsMerge does
  return pl_model, merged_lora, merged_classifier
"""