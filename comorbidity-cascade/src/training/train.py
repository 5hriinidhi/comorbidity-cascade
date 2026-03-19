import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ComorbidityDataset, create_kfold_splits, get_dataloaders
from models.mtl_flat import MTLFlat, masked_bce_loss
from models.graph_propagation import ComorbidityDAG
from models.mtl_graph import MTLWithGraph
from models.causal_loss import CausalConsistencyLoss

def get_model(model_name, input_dim, disease_names, config):
    if model_name == "mtl_flat":
        encoder_config = config['model']
        encoder_config['hidden_dims'] = config['model']['encoder_dims']
        head_config = {'hidden_dim': config['model']['task_head_dims'][0]}
        return MTLFlat(input_dim, disease_names, encoder_config, head_config)
    elif model_name in ["mtl_graph", "mtl_full"]:
        encoder_config = config['model']
        encoder_config['hidden_dims'] = config['model']['encoder_dims']
        head_config = {'hidden_dim': config['model']['task_head_dims'][0]}
        dag = ComorbidityDAG("config.yaml")
        return MTLWithGraph(input_dim, disease_names, dag, encoder_config, head_config)
    else:
        raise ValueError(f"Model {model_name} not implemented in trainer yet.")

def compute_class_weights(labels_df, disease_names):
    weights = []
    for d in disease_names:
        n_pos = labels_df[d].sum()
        n_neg = (labels_df[d] == 0).sum()
        if n_pos > 0:
            weight = n_neg / n_pos
        else:
            weight = 1.0
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, scheduler, device, class_weights, causal_loss_fn=None):
    model.train()
    total_loss = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        
        bce_loss = masked_bce_loss(y_pred, y, mask, class_weights.to(device))
        
        if causal_loss_fn:
            causal_loss = causal_loss_fn(y_pred)
            loss = bce_loss + causal_loss
        else:
            loss = bce_loss
            
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, disease_names):
    model.eval()
    all_preds = []
    all_trues = []
    all_masks = []
    
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            y_pred = model(x)
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    aurocs = []
    for i, _ in enumerate(disease_names):
        mask_i = all_masks[:, i] == 1.0
        if mask_i.sum() > 0 and len(np.unique(all_trues[mask_i, i])) > 1:
            auroc = roc_auc_score(all_trues[mask_i, i], all_preds[mask_i, i])
        else:
            auroc = 0.5
        aurocs.append(auroc)
        
    return aurocs, np.mean(aurocs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mtl_flat", "mtl_graph", "mtl_full"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    base_dir = "."
    features_df = pd.read_csv(os.path.join(base_dir, "data/final/features.csv"))
    labels_df = pd.read_csv(os.path.join(base_dir, "data/final/labels.csv"))
    with open(os.path.join(base_dir, "data/final/feature_names.json"), "r") as f:
        feature_names = json.load(f)
        
    disease_names = config["diseases"]["labels"]
    input_dim = len(feature_names)
    n_folds = config["training"]["n_folds"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["early_stop_patience"]
    
    output_path = args.output if args.output else f"results/{args.model}_auroc.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    
    all_fold_results = []
    
    for fold_idx in range(n_folds):
        print(f"\n--- FOLD {fold_idx} ---")
        train_loader, val_loader = get_dataloaders(fold_idx, features_df, labels_df, feature_names, batch_size=batch_size)
        
        # Get train labels for class weights
        splits = create_kfold_splits(features_df, labels_df, n_folds=n_folds)
        train_idx, _ = splits[fold_idx]
        class_weights = compute_class_weights(labels_df.iloc[train_idx], disease_names)
        
        model = get_model(args.model, input_dim, disease_names, config).to(device)
        
        causal_loss_fn = None
        if args.model == "mtl_full":
            dag = ComorbidityDAG("config.yaml")
            lambda_val = config["training"].get("lambda_selected", 0.15)
            causal_loss_fn = CausalConsistencyLoss(dag, disease_names, lambda_weight=lambda_val)
            print(f"Using CausalConsistencyLoss with lambda={lambda_val}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine Annealing Schedule
        num_epochs = 200 # Max epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
        
        best_macro_auroc = 0.0
        epochs_no_improve = 0
        ckpt_path = os.path.join(base_dir, f"checkpoints/{args.model}_fold{fold_idx}.pt")
        
        for epoch in range(num_epochs):
            avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, class_weights, causal_loss_fn)
            aurocs, macro_auroc = evaluate(model, val_loader, device, disease_names)
            
            if macro_auroc > best_macro_auroc:
                best_macro_auroc = macro_auroc
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val Macro-AUROC: {macro_auroc:.4f}")
                
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Final evaluation on best model
        model.load_state_dict(torch.load(ckpt_path))
        best_aurocs, best_macro = evaluate(model, val_loader, device, disease_names)
        print(f"Fold {fold_idx} Best Val Macro-AUROC: {best_macro:.4f}")
        
        for i, d in enumerate(disease_names):
            all_fold_results.append({
                "fold": fold_idx,
                "disease": d,
                "auroc": best_aurocs[i]
            })
            
    # Save results
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(output_path, index=False)
    
    print("\n--- FINAL SUMMARY ---")
    summary = results_df.groupby("disease")["auroc"].agg(["mean", "std"])
    print(summary)
    
    macro_means = results_df.groupby("fold")["auroc"].mean()
    print(f"\nMean Macro-AUROC: {macro_means.mean():.4f} ± {macro_means.std():.4f}")

if __name__ == "__main__":
    main()
