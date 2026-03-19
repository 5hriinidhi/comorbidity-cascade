import torch
import numpy as np
import json

class InterventionSimulationEngine:
    """
    Simulates interventions by modifying input features and measuring changes in model predictions.
    
    Usage:
        engine = InterventionSimulationEngine(model, feature_names, disease_names, config)
        mean_delta, std_delta, deltas = engine.intervene_relative(X, "BMXBMI", -5.0)
    """
    def __init__(self, model, feature_names, disease_names, config):
        """
        Args:
            model: Trained PyTorch model (MTLWithGraph or MTLFull)
            feature_names: List of all feature names in the order they appear in X
            disease_names: List of disease names corresponding to model output columns
            config: Full configuration dictionary
        """
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.disease_names = disease_names
        self.config = config
        self.device = next(model.parameters()).device

    def _get_feature_idx(self, feature_name):
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in feature_names.")
        return self.feature_names.index(feature_name)

    def _to_tensor(self, X):
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).float().to(self.device)
        return X.float().to(self.device)

    def intervene(self, X, feature_name, new_value):
        """Sets a specific feature to a fixed value for all samples."""
        X_baseline = self._to_tensor(X)
        feature_idx = self._get_feature_idx(feature_name)
        
        X_intervened = X_baseline.clone()
        X_intervened[:, feature_idx] = new_value
        
        with torch.no_grad():
            p_baseline = self.model(X_baseline)
            p_intervened = self.model(X_intervened)
            
        deltas = p_intervened - p_baseline
        mean_delta = deltas.mean(dim=0).cpu().numpy()
        std_delta = deltas.std(dim=0).cpu().numpy()
        
        return mean_delta, std_delta, deltas.cpu().numpy()

    def intervene_relative(self, X, feature_name, delta_value):
        """Adjusts a specific feature by a delta amount per sample."""
        X_baseline = self._to_tensor(X)
        feature_idx = self._get_feature_idx(feature_name)
        
        X_intervened = X_baseline.clone()
        X_intervened[:, feature_idx] += delta_value
        
        with torch.no_grad():
            p_baseline = self.model(X_baseline)
            p_intervened = self.model(X_intervened)
            
        deltas = p_intervened - p_baseline
        mean_delta = deltas.mean(dim=0).cpu().numpy()
        std_delta = deltas.std(dim=0).cpu().numpy()
        
        return mean_delta, std_delta, deltas.cpu().numpy()

if __name__ == "__main__":
    import yaml
    import os
    import sys
    import pandas as pd
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.models.mtl_graph import MTLWithGraph
    from src.models.graph_propagation import ComorbidityDAG
    
    # Load config and names
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("data/final/feature_names.json", "r") as f:
        feature_names = json.load(f)
    disease_names = config["diseases"]["labels"]
    
    # Load trained model (Fold 0)
    input_dim = len(feature_names)
    dag = ComorbidityDAG("config.yaml")
    encoder_config = config['model']
    encoder_config['hidden_dims'] = config['model']['encoder_dims']
    head_config = {'hidden_dim': config['model']['task_head_dims'][0]}
    
    model = MTLWithGraph(input_dim, disease_names, dag, encoder_config, head_config)
    checkpoint_path = "checkpoints/mtl_full_fold0.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded trained model from {checkpoint_path}")
    else:
        print("WARNING: Using random weights (checkpoint not found)")

    engine = InterventionSimulationEngine(model, feature_names, disease_names, config)
    
    # Load real data samples
    features_df = pd.read_csv("data/final/features.csv")
    X_test = features_df[feature_names].values[:100]
    
    print("\n--- Test 1: Relative Intervention (BMI Reduction: -5.0) ---")
    mean_delta, _, per_sample = engine.intervene_relative(X_test, "BMXBMI", -5.0)
    
    # Precision threshold for floating point
    eps = 1e-4
    all_non_positive = np.all(per_sample <= eps)
    
    print(f"Mean Delta per disease:")
    for d, m in zip(disease_names, mean_delta):
        print(f"  {d:<15}: {m:+.6f}")
        
    print(f"\nAll per-sample deltas <= {eps}: {all_non_positive}")
    
    if not all_non_positive:
        max_violation = per_sample.max()
        print(f"FAILED: Found positive delta of {max_violation}. Check model training or sign of BMI feature.")
    else:
        print("PASSED: BMI reduction consistently decreases or maintains risk.")
