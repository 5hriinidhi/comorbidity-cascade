import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import create_kfold_splits, get_dataloaders
from src.models.mtl_graph import MTLWithGraph
from src.models.graph_propagation import ComorbidityDAG
from src.models.causal_loss import CausalConsistencyLoss
from src.models.mtl_flat import masked_bce_loss
from src.training.train import compute_class_weights, evaluate

def calculate_causal_inversion_rate(model, loader, device, dag, disease_names):
    model.eval()
    inversions = 0
    total_samples = 0
    
    # Pre-compute edge indices
    edge_indices = []
    for src, dst, weight in dag.edges:
        src_idx = disease_names.index(src)
        dst_idx = disease_names.index(dst)
        edge_indices.append((src_idx, dst_idx))
        
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y_pred = model(x) # (batch, 7)
            
            # For each sample in batch
            for b in range(y_pred.shape[0]):
                has_inversion = False
                for src_idx, dst_idx in edge_indices:
                    if y_pred[b, dst_idx] > y_pred[b, src_idx]:
                        has_inversion = True
                        break
                if has_inversion:
                    inversions += 1
                total_samples += 1
                
    return (inversions / total_samples) * 100 if total_samples > 0 else 0.0

def train_full_model(lambda_val, fold_idx, features_df, labels_df, feature_names, disease_names, config, device):
    print(f"\n--- Training Full Model with lambda={lambda_val} on Fold {fold_idx} ---")
    
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["early_stop_patience"]
    num_epochs = 100 # Reduced epochs for faster hparam search
    
    train_loader, val_loader = get_dataloaders(fold_idx, features_df, labels_df, feature_names, batch_size=batch_size)
    
    # Class weights
    splits = create_kfold_splits(features_df, labels_df, n_folds=config["training"]["n_folds"])
    train_idx, _ = splits[fold_idx]
    class_weights = compute_class_weights(labels_df.iloc[train_idx], disease_names).to(device)
    
    # Model components
    dag = ComorbidityDAG("config.yaml")
    encoder_config = config['model']
    encoder_config['hidden_dims'] = config['model']['encoder_dims']
    head_config = {'hidden_dim': config['model']['task_head_dims'][0]}
    
    input_dim = len(feature_names)
    model = MTLWithGraph(input_dim, disease_names, dag, encoder_config, head_config).to(device)
    
    causal_loss_fn = CausalConsistencyLoss(dag, disease_names, lambda_weight=lambda_val)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    
    best_macro_auroc = 0.0
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            
            bce = masked_bce_loss(y_pred, y, mask, class_weights)
            causal = causal_loss_fn(y_pred)
            loss = bce + causal
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        aurocs, macro_auroc = evaluate(model, val_loader, device, disease_names)
        
        if macro_auroc > best_macro_auroc:
            best_macro_auroc = macro_auroc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    _, final_macro_auroc = evaluate(model, val_loader, device, disease_names)
    inversion_rate = calculate_causal_inversion_rate(model, val_loader, device, dag, disease_names)
    
    return final_macro_auroc, inversion_rate

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    features_df = pd.read_csv("data/final/features.csv")
    labels_df = pd.read_csv("data/final/labels.csv")
    with open("data/final/feature_names.json", "r") as f:
        feature_names = json.load(f)
        
    disease_names = config["diseases"]["labels"]
    lambdas = [0.05, 0.10, 0.15, 0.20, 0.25]
    fold_idx = 0
    
    results = []
    
    for l in lambdas:
        auroc, inv_rate = train_full_model(l, fold_idx, features_df, labels_df, feature_names, disease_names, config, device)
        results.append({
            "lambda": l,
            "val_macro_auroc": auroc,
            "causal_inversion_rate_%": inv_rate
        })
        
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/hparam_search_log.csv", index=False)
    
    print("\n--- Hyperparameter Search Results ---")
    print(results_df.to_string(index=False))
    
    # Selection logic: max AUROC while inversion rate < 10%
    valid_results = results_df[results_df["causal_inversion_rate_%"] < 10.0]
    if not valid_results.empty:
        selected_row = valid_results.loc[valid_results["val_macro_auroc"].idxmax()]
        selected_lambda = selected_row["lambda"]
    else:
        # Fallback to minimum inversion rate if none are < 10%
        selected_row = results_df.loc[results_df["causal_inversion_rate_%"].idxmin()]
        selected_lambda = selected_row["lambda"]
        
    print(f"\nSELECTED lambda = {selected_lambda}")
    
    # Save back to config.yaml
    config["training"]["lambda_selected"] = float(selected_lambda)
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print("Updated config.yaml with training.lambda_selected")

if __name__ == "__main__":
    main()
