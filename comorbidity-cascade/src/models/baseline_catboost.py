import os
import yaml
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import sys

# Ensure we can import from src.data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import create_kfold_splits

class IndependentCatBoostBaseline:
    def __init__(self, disease_names, config):
        self.disease_names = disease_names
        self.config = config
        self.models = {}
        self.skipped_diseases = []

    def fit(self, X_train, y_train):
        for disease in self.disease_names:
            # Drop NaN labels for this specific disease
            mask = y_train[disease].notna()
            X_disease = X_train[mask]
            y_disease = y_train[disease][mask]
            
            n_pos = (y_disease == 1).sum()
            n_neg = (y_disease == 0).sum()
            
            if n_pos < 10:
                print(f"Skipping disease {disease} - insufficient positive samples ({n_pos})")
                self.skipped_diseases.append(disease)
                continue
            
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            
            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=False,
                scale_pos_weight=scale_pos_weight,
                random_seed=42
            )
            
            model.fit(X_disease, y_disease)
            self.models[disease] = model

    def predict_proba(self, X):
        n_samples = len(X)
        n_diseases = len(self.disease_names)
        probas = np.zeros((n_samples, n_diseases))
        
        for i, disease in enumerate(self.disease_names):
            if disease in self.models:
                # CatBoost predict_proba returns [prob_0, prob_1]
                probas[:, i] = self.models[disease].predict_proba(X)[:, 1]
            else:
                probas[:, i] = 0.5
                
        return probas

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for disease, model in self.models.items():
            model.save_model(os.path.join(path, f"{disease}.cbm"))
        
        meta = {
            "disease_names": self.disease_names,
            "skipped_diseases": self.skipped_diseases
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)
        self.disease_names = meta["disease_names"]
        self.skipped_diseases = meta["skipped_diseases"]
        
        for disease in self.disease_names:
            if disease not in self.skipped_diseases:
                model = CatBoostClassifier()
                model.load_model(os.path.join(path, f"{disease}.cbm"))
                self.models[disease] = model

if __name__ == "__main__":
    # Load config
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Load data
    features_df = pd.read_csv(os.path.join(base_dir, "data/final/features.csv"))
    labels_df = pd.read_csv(os.path.join(base_dir, "data/final/labels.csv"))
    
    with open(os.path.join(base_dir, "data/final/feature_names.json"), "r") as f:
        feature_names = json.load(f)
        
    disease_names = config["diseases"]["labels"]
    n_folds = config["training"]["n_folds"]
    
    # Initialize results
    all_results = []
    
    # K-Fold splits
    splits = create_kfold_splits(features_df, labels_df, n_folds=n_folds, seed=42)
    
    print(f"Starting {n_folds}-fold CV for CatBoost Baseline...")
    
    fold_macro_aurocs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"--- Fold {fold_idx} ---")
        
        X_train = features_df.iloc[train_idx][feature_names]
        X_val = features_df.iloc[val_idx][feature_names]
        y_train = labels_df.iloc[train_idx]
        y_val = labels_df.iloc[val_idx]
        
        baseline = IndependentCatBoostBaseline(disease_names, config)
        baseline.fit(X_train, y_train)
        
        y_probs = baseline.predict_proba(X_val)
        
        fold_aurocs = []
        for i, disease in enumerate(disease_names):
            y_true = y_val[disease]
            # Use only samples with known labels
            mask = y_true.notna()
            if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
                auroc = roc_auc_score(y_true[mask], y_probs[mask, i])
            else:
                auroc = 0.5
            
            fold_aurocs.append(auroc)
            all_results.append({
                "fold": fold_idx,
                "disease": disease,
                "auroc": auroc
            })
            print(f"  {disease:<15} AUROC: {auroc:.4f}")
            
        macro_auroc = np.mean(fold_aurocs)
        fold_macro_aurocs.append(macro_auroc)
        print(f"  Macro-AUROC: {macro_auroc:.4f}")
        
    print("\n--- FINAL SUMMARY ---")
    avg_macro_auroc = np.mean(fold_macro_aurocs)
    print(f"Average Macro-AUROC across {n_folds} folds: {avg_macro_auroc:.4f}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(base_dir, "results/baseline_auroc.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
