import sys
import os
import torch

# Add src to the path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.graph_propagation import ComorbidityDAG, build_augmented_inputs

def test_dag():
    print("--- TESTING ComorbidityDAG ---")
    config_path = 'config.yaml'
    dag = ComorbidityDAG(config_path)
    
    # Verify Topological Order
    topo_order = dag.topological_order()
    print(f"Topological Order: {topo_order}")
    
    # Root check
    roots = ['obesity', 'osteoporosis']
    for root in roots:
        if root in topo_order:
            idx = topo_order.index(root)
            print(f"{root} index: {idx}")
    
    # Predecessor check for ckd
    ckd_preds = dag.get_predecessors('ckd')
    print(f"ckd predecessors: {ckd_preds}")
    
    # Predecessor check for stroke
    stroke_preds = dag.get_predecessors('stroke')
    print(f"stroke predecessors: {stroke_preds}")
    
    # Predecessor check for obesity (root)
    obesity_preds = dag.get_predecessors('obesity')
    print(f"obesity predecessors: {obesity_preds}")
    
    # Predecessor check for osteoporosis (no predecessors)
    osteo_preds = dag.get_predecessors('osteoporosis')
    print(f"osteoporosis predecessors: {osteo_preds}")

def test_augmented_inputs():
    print("\n--- TESTING build_augmented_inputs ---")
    config_path = 'config.yaml'
    dag = ComorbidityDAG(config_path)
    disease_names = ['obesity', 't2d', 'hypertension', 'cad', 'ckd', 'stroke', 'osteoporosis']
    
    batch_size = 8
    hidden_dim = 128
    
    # Dummy embedding
    z = torch.randn(batch_size, hidden_dim)
    
    # Dummy initial preds
    initial_preds = {
        d: torch.sigmoid(torch.randn(batch_size, 1)) for d in disease_names
    }
    
    # Build augmented inputs
    augmented_inputs = build_augmented_inputs(z, initial_preds, dag, disease_names)
    
    print(f"Number of disease entries in output: {len(augmented_inputs)}")
    
    for d, z_tilde in augmented_inputs.items():
        preds = dag.get_predecessors(d)
        num_preds = len(preds)
        expected_dim = hidden_dim + num_preds
        print(f"Disease: {d:15} | num_preds: {num_preds} | shape: {z_tilde.shape} | Expected dim: {expected_dim}")
        
        assert z_tilde.shape == (batch_size, expected_dim), f"Shape mismatch for {d}!"

if __name__ == "__main__":
    test_dag()
    test_augmented_inputs()
    print("\n[VERIFICATION SUCCESSFUL]")
