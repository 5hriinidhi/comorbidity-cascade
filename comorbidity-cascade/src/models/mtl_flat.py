import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.encoder import SharedEncoder
from src.models.task_heads import DiseaseTaskHead

class MTLFlat(nn.Module):
    def __init__(self, input_dim, disease_names, encoder_config, head_config):
        super(MTLFlat, self).__init__()
        self.disease_names = disease_names
        
        # Shared Encoder
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_config.get('hidden_dims', [256, 128]),
            dropout=encoder_config.get('dropout', 0.30),
            batch_norm=encoder_config.get('batch_norm', True)
        )
        
        # Task Heads
        encoder_out_dim = encoder_config.get('hidden_dims', [256, 128])[-1]
        self.heads = nn.ModuleDict({
            disease: DiseaseTaskHead(
                input_dim=encoder_out_dim,
                hidden_dim=head_config.get('hidden_dim', 64),
                disease_name=disease
            ) for disease in disease_names
        })
        
    def forward(self, x):
        """Returns predictions of shape (batch, 7)"""
        z = self.encoder(x)
        # Concatenate predictions from each head
        preds = [self.heads[d](z) for d in self.disease_names]
        return torch.cat(preds, dim=1)

def masked_bce_loss(y_pred, y_true, mask, class_weights=None):
    """
    Computes masked multi-label binary cross entropy loss.
    y_pred: (batch, 7)
    y_true: (batch, 7)
    mask: (batch, 7)
    class_weights: (7,)
    """
    # Binary Cross Entropy with reduction='none' to apply mask and weights
    loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    
    # Apply label masking
    masked_loss = loss * mask
    
    # Apply class weights if provided
    if class_weights is not None:
        # Reshape class_weights to (1, 7) for broadcasting
        weights = class_weights.view(1, -1)
        masked_loss = masked_loss * weights
        
    # Average across non-masked entries
    num_non_masked = mask.sum()
    if num_non_masked == 0:
        return torch.tensor(0.0, requires_grad=True, device=y_pred.device)
        
    return masked_loss.sum() / num_non_masked

if __name__ == "__main__":
    import torch
    print("Testing MTLFlat and masked_bce_loss...")
    
    input_dim = 30
    disease_names = ["obesity", "t2d", "hypertension", "cad", "ckd", "stroke", "osteoporosis"]
    encoder_config = {'hidden_dims': [256, 128], 'dropout': 0.3, 'batch_norm': True}
    head_config = {'hidden_dim': 64}
    
    model = MTLFlat(input_dim, disease_names, encoder_config, head_config)
    
    # Test forward pass
    x = torch.randn(32, input_dim)
    y_pred = model(x)
    assert y_pred.shape == (32, 7), f"Expected (32, 7), got {y_pred.shape}"
    assert (y_pred >= 0).all() and (y_pred <= 1).all(), "Predictions must be in [0, 1]"
    
    # Test loss function
    y_true = torch.zeros(32, 7)
    mask = torch.ones(32, 7)
    
    # All ones mask
    loss_all = masked_bce_loss(y_pred, y_true, mask)
    assert loss_all > 0, "Loss should be positive when mask is all ones"
    print(f"Loss (all ones mask): {loss_all.item():.4f}")
    
    # All zeros mask
    mask_zeros = torch.zeros(32, 7)
    loss_zero = masked_bce_loss(y_pred, y_true, mask_zeros)
    assert loss_zero.item() == 0.0, f"Loss should be 0.0 when mask is all zeros, got {loss_zero.item()}"
    
    print("MTLFlat and masked_bce_loss OK")
