import torch
import torch.nn as nn

class SharedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.30, batch_norm=True):
        super(SharedEncoder, self).__init__()
        
        layers = []
        curr_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(curr_dim, h_dim))
            
            # Batch Normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
                
            # ReLU activation
            layers.append(nn.ReLU())
            
            # Update current dimension
            curr_dim = h_dim
            
        # Final Dropout
        layers.append(nn.Dropout(p=dropout))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """Returns z of shape (batch, 128)"""
        return self.encoder(x)

if __name__ == "__main__":
    # Standalone test
    print("Testing SharedEncoder...")
    
    input_dim = 30
    encoder = SharedEncoder(input_dim=input_dim)
    
    # Verify shape
    x = torch.randn(32, input_dim)
    z = encoder(x)
    assert z.shape == (32, 128), f"Expected (32, 128), got {z.shape}"
    
    # Verify train/eval modes
    encoder.train()
    assert encoder.training == True, "Expected encoder.training to be True in train mode"
    
    encoder.eval()
    assert encoder.training == False, "Expected encoder.training to be False in eval mode"
    
    # Test output consistency
    z_eval = encoder(x)
    assert z_eval.shape == (32, 128), f"Expected (32, 128) in eval mode, got {z_eval.shape}"
    
    print("SharedEncoder OK")
