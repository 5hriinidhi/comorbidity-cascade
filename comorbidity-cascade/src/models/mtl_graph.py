import os
import sys
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.encoder import SharedEncoder
from src.models.task_heads import DiseaseTaskHead
from src.models.graph_propagation import build_augmented_inputs

class MTLWithGraph(nn.Module):
    def __init__(self, input_dim, disease_names, dag, encoder_config, head_config):
        """
        MTL model with Graph Propagation (two-pass architecture).
        
        Args:
            input_dim: Number of input features
            disease_names: List of disease names
            dag: ComorbidityDAG instance
            encoder_config: Dict with 'hidden_dims', 'dropout', 'batch_norm'
            head_config: Dict with 'hidden_dim'
        """
        super(MTLWithGraph, self).__init__()
        self.disease_names = disease_names
        self.dag = dag
        
        # Shared Encoder
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_config.get('hidden_dims', [256, 128]),
            dropout=encoder_config.get('dropout', 0.30),
            batch_norm=encoder_config.get('batch_norm', True)
        )
        
        encoder_out_dim = encoder_config.get('hidden_dims', [256, 128])[-1]
        
        # Task Heads: 
        # root nodes: input_dim = 128
        # nodes with k predecessors: input_dim = 128 + k
        self.heads = nn.ModuleDict()
        for disease in disease_names:
            preds = dag.get_predecessors(disease)
            k = len(preds)
            head_input_dim = encoder_out_dim + k
            
            self.heads[disease] = DiseaseTaskHead(
                input_dim=head_input_dim,
                hidden_dim=head_config.get('hidden_dim', 64),
                disease_name=disease
            )
            print(f"[MTL-GRAPH] Head: {disease:15} | input_dim: {head_input_dim}")

    def forward(self, x):
        """
        Two-pass forward logic:
        Pass 1: Use plain z to get initial predictions for each disease.
        Pass 2: Use graph propagation to augment z with weighted initial predictions of predecessors.
        """
        # SHARED ENCODING
        z = self.encoder(x) # (batch, 128)
        
        # PASS 1: Initial rough predictions for ALL diseases
        # These are used to provide 'upstream' context for the second pass.
        initial_preds = {}
        for d in self.disease_names:
            # head_d.layer1 maps from 128 -> 1
            initial_preds[d] = torch.sigmoid(self.heads[d].layer1(z))
            
        # PASS 2: Augmented pass using graph propagation
        # returns dict: disease -> z_tilde (concat of z and weighted upstream preds)
        z_tildes = build_augmented_inputs(z, initial_preds, self.dag, self.disease_names)
        
        final_preds = {}
        for d in self.disease_names:
            # head_d.forward uses head_d.net (maps from 128+k -> 1)
            final_preds[d] = self.heads[d](z_tildes[d])
            
        # Return concatenated predictions: (batch, n_diseases)
        return torch.cat([final_preds[d] for d in self.disease_names], dim=1)

if __name__ == "__main__":
    # Internal test
    import os
    import yaml
    from src.models.graph_propagation import ComorbidityDAG
    
    print("\n--- TESTING MTLWithGraph ---")
    
    # Load config and DAG
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        # Create a mock config for standalone test if needed, 
        # but here we assume we are running from project root.
        pass
        
    dag = ComorbidityDAG(config_path)
    disease_names = ["obesity", "t2d", "hypertension", "cad", "ckd", "stroke", "osteoporosis"]
    
    input_dim = 30
    encoder_config = {'hidden_dims': [256, 128], 'dropout': 0.3, 'batch_norm': True}
    head_config = {'hidden_dim': 64}
    
    model = MTLWithGraph(input_dim, disease_names, dag, encoder_config, head_config)
    
    # Test forward pass
    x = torch.randn(32, input_dim)
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (32, 7), f"Expected (32, 7), got {output.shape}"
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("Backward pass successful")
    
    print("MTLWithGraph OK")
