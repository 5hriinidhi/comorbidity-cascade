import torch
import torch.nn as nn

class DiseaseTaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, disease_name=''):
        super(DiseaseTaskHead, self).__init__()
        self.disease_name = disease_name
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        """Returns scalar probability in [0,1], shape (batch, 1)"""
        return self.net(z)

if __name__ == "__main__":
    print("Testing DiseaseTaskHead...")
    batch_size = 32
    input_dim = 128
    head = DiseaseTaskHead(input_dim=input_dim, disease_name='obesity')
    z = torch.randn(batch_size, input_dim)
    output = head(z)
    
    assert output.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {output.shape}"
    assert (output >= 0).all() and (output <= 1).all(), "Output values must be in [0, 1]"
    
    print("DiseaseTaskHead OK")
