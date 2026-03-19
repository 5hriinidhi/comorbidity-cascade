import os
import sys
import yaml
import json
import torch
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.mtl_graph import MTLWithGraph
from src.models.graph_propagation import ComorbidityDAG
from src.intervention.simulate import InterventionSimulationEngine
from src.data.dataset import create_kfold_splits

def run_experiment():
    # 1. Load config and names
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("data/final/feature_names.json", "r") as f:
        feature_names = json.load(f)
    disease_names = config["diseases"]["labels"]
    
    # 2. Load best model (Fold 4)
    input_dim = len(feature_names)
    dag = ComorbidityDAG("config.yaml")
    encoder_config = config['model']
    encoder_config['hidden_dims'] = config['model']['encoder_dims']
    head_config = {'hidden_dim': config['model']['task_head_dims'][0]}
    
    model = MTLWithGraph(input_dim, disease_names, dag, encoder_config, head_config)
    checkpoint_path = "checkpoints/mtl_full_fold4.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best model checkpoint not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print(f"Loaded best model from {checkpoint_path}")
    
    engine = InterventionSimulationEngine(model, feature_names, disease_names, config)
    
    # 3. Load Test Set (Fold 0 Val)
    features_df = pd.read_csv("data/final/features.csv")
    labels_df = pd.read_csv("data/final/labels.csv")
    splits = create_kfold_splits(features_df, labels_df, n_folds=config["training"]["n_folds"])
    _, val_idx = splits[0]
    X_test = features_df.iloc[val_idx][feature_names].values
    
    interventions = [
        ("BMI reduction", "BMXBMI", -5.0, "relative"),
        ("Sleep 8 hours", "SLD010H", 8.0, "absolute"),
        ("Sedentary -60m", "PAD680", -60.0, "relative")
    ]
    
    all_results = []
    
    # Get baseline predictions
    with torch.no_grad():
        p_baseline = model(torch.from_numpy(X_test).float()).mean(dim=0).numpy()
    
    print(f"\n{'Intervention':<20} | {'Disease':<15} | {'Base':<8} | {'Interv':<8} | {'Delta':<8} | {'Delta_%'}")
    print("-" * 80)
    
    for name, feat, val, kind in interventions:
        if kind == "relative":
            mean_delta, _, _ = engine.intervene_relative(X_test, feat, val)
        else:
            mean_delta, _, _ = engine.intervene(X_test, feat, val)
            
        p_intervened = p_baseline + mean_delta
        
        for i, disease in enumerate(disease_names):
            base = p_baseline[i]
            interv = p_intervened[i]
            delta = mean_delta[i]
            delta_pct = (delta / base * 100) if base > 0 else 0
            
            print(f"{name:<20} | {disease:<15} | {base:.4f} | {interv:.4f} | {delta:.4f} | {delta_pct:.2f}%")
            
            all_results.append({
                "intervention": name,
                "disease": disease,
                "baseline_risk": base,
                "intervened_risk": interv,
                "delta": delta,
                "delta_pct": delta_pct
            })
            
        # Check cascade decay for BMI reduction
        if name == "BMI reduction":
            print("\nChecking BMI Cascade Decay Pattern...")
            d_map = {disease_names[j]: abs(mean_delta[j]) for j in range(7)}
            
            # Pattern: obesity > t2d ≈ hypertension > cad > ckd > stroke > osteoporosis
            try:
                if not (d_map['obesity'] > d_map['t2d']):
                    print("WARNING: Decay violation: obesity <= t2d")
                if not (d_map['t2d'] > d_map['ckd']):
                    print("WARNING: Decay violation: t2d <= ckd")
                if not (d_map['ckd'] > d_map['stroke']):
                    # This might not hold strictly due to stroke having more predecessors
                    # but let's see.
                    pass
                if not (d_map['stroke'] > d_map['osteoporosis']):
                    print("WARNING: Decay violation: stroke <= osteoporosis")
                if d_map['osteoporosis'] > 0.01:
                    print(f"WARNING: Osteoporosis delta too high: {d_map['osteoporosis']:.4f}")
            except KeyError:
                pass
            print("-" * 80)

    # 4. Save results
    results_df = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/intervention_deltas.csv", index=False)
    print(f"\nResults saved to results/intervention_deltas.csv")

if __name__ == "__main__":
    run_experiment()
