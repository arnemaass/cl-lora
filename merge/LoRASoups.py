import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file



def LoRASoupsMerge(pl_model, train_loader, val_loader, lora_head1, lora_head2, fc_head1, fc_head2, mode = 'learnable'):
    '''
    Apply LoraSoups by merging LoRA weights from multiple models.

    
    1. Take the base model as input and freeze all parameters
    2. Load the LoRA Heads
    3. Initialze as many alpha values as LoRA heads
    4. Add the additional LoRA Heads mutlpiplied by their alpha coeffiecients to the representive layer of the base model. 
    5. If mode = learnable: set the alpha values as trainable, else alpha = 1/n, where n is the number of LoRA heads.
    6. Unfreeze the classification head
    7. Train the alpha values together with the classification head 
    8. Return the merged model, LoRA weights and classifier weights

    The update per layer l is computed as:
    \Delta W^l = \alpha_1^l B_1A_1^T + \alpha_2^l B_2A_2^T
    = alpha B[B1;B2] @ A[A1;A2]'
    Delta Matrix = B_concat' @ alpha_block @ A_concat

    mode = 'static':
        - Simply use equal weights for the LoRA Heads (e.g., if n = 2: \alpha = 0.5)

    mode = 'learnable':
        - Freeze all base and LoRA weights
        - Define per-layer trainable coefficients \alpha_i^l for each LoRA head i
        - At each layer l, compute the LoRA update as:
            \Delta W^l = \sum_i \alphai^l \cdot (Bi^l @ Ai^l)^T
        - Learn \alpha via gradient descent on a small held-out set

    Args:
        pl_model: PyTorch Lightning model (base model)
        train_loader: Training data loader
        val_loader: Validation data loader
        lora_heads: List of LoRA weight dictionaries for each task
        classifier_heads: List of classifier weight dictionaries for each task
        mode: 'learnable' or 'static'
        num_epochs: Number of training epochs (default: 1)
        lr: Learning rate (default: 1e-4)

    Returns:
        Tuple of (merged_model, merged_lora_weights, merged_classifier_weights)
        
    '''

    # 1. Freeze all base model parameters
    for param in pl_model.parameters():
        param.requires_grad = False
    
    # 2. Extract LoRA layers and organize by layer index
    num_heads = len(lora_heads)
    
    # Find all layer indices from the first LoRA head
    layer_indices = set()
    for key in lora_heads[0].keys():
        if key.startswith('w_a_') or key.startswith('w_b_'):
            layer_idx = key.split('_')[-1]  # Extract XXX from w_a_XXX or w_b_XXX
            layer_indices.add(layer_idx)
    
    layer_indices = sorted(list(layer_indices))
    num_layers = len(layer_indices)
    
    # 3. Initialize alpha values
    if mode == 'learnable':
        # Trainable alpha parameters per layer per head
        alpha_params = nn.ParameterDict()
        for layer_idx in layer_indices:
            alpha_params[f'layer_{layer_idx}'] = nn.Parameter(
                torch.ones(num_heads) / num_heads
            )
    else:  # static mode
        # Equal weights: alpha = 1/n
        static_alpha = 1.0 / num_heads
        alpha_params = {
            f'layer_{layer_idx}': torch.full((num_heads,), static_alpha)
            for layer_idx in layer_indices
        }
    
    # 4. Merge LoRA weights and apply to base model
    merged_lora_weights = {}
    
    for layer_idx in layer_indices:
        layer_key = f'layer_{layer_idx}'
        
        if mode == 'learnable':
            alphas = F.softmax(alpha_params[layer_key], dim=0)
        else:
            alphas = alpha_params[layer_key]
        
        # Collect all B and A matrices for this layer
        B_matrices = []
        A_matrices = []
        
        for head_idx, lora_head in enumerate(lora_heads):
            a_key = f'w_a_{layer_idx}'
            b_key = f'w_b_{layer_idx}'
            
            if a_key in lora_head and b_key in lora_head:
                A = lora_head[a_key]  # Shape: [rank, in_features]
                B = lora_head[b_key]  # Shape: [out_features, rank]
                
                B_matrices.append(B)
                A_matrices.append(A)
        
        if B_matrices and A_matrices:
            # Concatenate B matrices: B[B1;B2] -> [out_features, total_rank]
            B_concat = torch.cat(B_matrices, dim=1)  # Concatenate along rank dimension
            
            # Concatenate A matrices: A[A1;A2] -> [total_rank, in_features]
            A_concat = torch.cat(A_matrices, dim=0)  # Concatenate along rank dimension
            
            # Create alpha weighting matrix - block diagonal to weight each head's contribution
            rank_per_head = A_matrices[0].shape[0]  # Assuming same rank for all heads
            alpha_diag = []
            for head_idx, alpha in enumerate(alphas):
                alpha_diag.append(alpha * torch.eye(rank_per_head, device=A_concat.device))
            
            # Create block diagonal alpha matrix
            alpha_block = torch.block_diag(*alpha_diag)  # Shape: [total_rank, total_rank]
            
            # Compute: B_concat @ alpha_block @ A_concat^T
            # This implements: W_delta = B[B1;B2] @ alpha @ A[A1;A2]'
            lora_update = B_concat @ alpha_block @ A_concat  # Shape: [out_features, in_features]
            
            # Store merged weights
            merged_lora_weights[layer_idx] = {
                'w_a': A_concat,
                'w_b': B_concat,
                'alpha_block': alpha_block,
                'update': lora_update
            }
            # TODO 
            # Apply LoRA update to the corresponding layer in the base model
            try:
                layer_name = f'blocks.{layer_idx}.attn.qkv' 
                target_layer = pl_model
                for part in layer_name.split('.'):
                    target_layer = getattr(target_layer, part)
                
                if hasattr(target_layer, 'weight'):
                    target_layer.weight.data += lora_update
            except AttributeError:
                print(f"Could not find layer {layer_name} in model")
                continue
    
    # 5. Merge classifier heads
    merged_classifier = {}
    
    for head_idx, classifier_head in enumerate(classifier_heads):
        weight = 1.0 / num_heads  # Equal weighting
        
        # Look for classifier weights (fc_XXXin_XXXout pattern)
        for param_name, param_tensor in classifier_head.items():
            if param_name.startswith('fc_'):
                if param_name not in merged_classifier:
                    merged_classifier[param_name] = weight * param_tensor
                else:
                    merged_classifier[param_name] += weight * param_tensor
    
    # Apply merged classifier weights
    classifier = None
    if hasattr(pl_model, 'classifier'):
        classifier = pl_model.classifier
    elif hasattr(pl_model, 'head'):
        classifier = pl_model.head
    elif hasattr(pl_model, 'fc'):
        classifier = pl_model.fc
    
    if classifier is not None and merged_classifier:
        # Apply the merged classifier weights
        for param_name, param_tensor in merged_classifier.items():
            if hasattr(classifier, 'weight'):
                classifier.weight.data = param_tensor
    
    # 6. Unfreeze classification head
    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True
    
    # 7. Training loop for learnable mode
    if mode == 'learnable':
        trainable_params = []
        
        # Add alpha parameters
        for param in alpha_params.values():
            if isinstance(param, nn.Parameter):
                trainable_params.append(param)
        
        # Add classifier parameters
        if classifier is not None:
            trainable_params.extend(classifier.parameters())
        
        if trainable_params:
            optimizer = torch.optim.Adam(trainable_params, lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            pl_model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    # Handle different batch formats
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        inputs = batch['image'] if isinstance(batch, dict) else batch
                        targets = batch['label'] if isinstance(batch, dict) else None
                    
                    if targets is None:
                        continue
                    
                    outputs = pl_model(inputs)
                    loss = criterion(outputs, targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # 8. Return merged model, LoRA weights, and classifier weights
    return pl_model, merged_lora_weights, merged_classifier


class AlphaMerger(nn.Module):
    """
    Returns softmax-normalized per-layer, per-head alpha coefficients.
    """
    def __init__(self, num_layers: int, num_heads: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_layers, num_heads) / num_heads)

    def forward(self):
        return F.softmax(self.alpha, dim=-1)  # (num_layers, num_heads)


class LayerMerger(nn.Module):
    """
    Merges LoRA updates from multiple heads into a base model.
    """
    def __init__(self, base_weights: torch.Tensor, num_heads: int, num_layers: int, learnable: bool = True):
        super().__init__()
        self.base_weights = base_weights  # Tensor of shape (num_layers, ...)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learnable = learnable

        # Learnable alpha parameters for weighted merging
        init_alpha = torch.ones(num_layers, num_heads) / num_heads
        self.alpha = nn.Parameter(init_alpha) if learnable else init_alpha

    def forward(self, lora_heads: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lora_heads: Tensor (num_layers, num_heads, B_dim, A_dim)

        Returns:
            merged_weights: Tensor (num_layers, B_dim, A_dim)
        """
        if self.learnable:
            alphas = F.softmax(self.alpha, dim=-1)
        else:
            alphas = self.alpha  # Static equal weighting

        merged_weights = []
        for l in range(self.num_layers):
            lora_update = sum(alphas[l, i] * lora_heads[l, i] for i in range(self.num_heads))
            merged_layer = self.base_weights[l] + lora_update
            merged_weights.append(merged_layer)

        return torch.stack(merged_weights, dim=0)