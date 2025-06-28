import torch
import torch.nn as nn
import torch.nn.functional as F



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

    mode = 'static':
        - Simply use equal weights for the LoRA Heads (e.g., if n = 2: \alpha = 0.5)

    mode = 'learnable':
        - Freeze all base and LoRA weights
        - Define per-layer trainable coefficients \alpha_i^l for each LoRA head i
        - At each layer l, compute the LoRA update as:
            \Delta W^l = \sum_i \alpha_i^l \cdot (B_i^l @ A_i^l)^T
        - Learn \alpha via gradient descent on a small held-out set
        
    '''

    for param in pl_model.parameters():
        param.requires_grad = False

    for adapter in pl_model:
        for param in adapter.parameters():
            param.requires_grad = False


    pass



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