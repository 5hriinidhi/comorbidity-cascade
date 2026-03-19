import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
from skmultilearn.model_selection import IterativeStratification

class ComorbidityDataset(Dataset):
    def __init__(self, features_df, labels_df, feature_names):
        disease_labels = ["obesity", "t2d", "hypertension", "cad", "ckd", "stroke", "osteoporosis"]
        self.X = features_df[feature_names].values.astype(np.float32)
        
        y_df = labels_df[disease_labels].copy()
        
        # Mask is 1.0 where not nan, 0.0 where nan
        self.mask = y_df.notna().values.astype(np.float32)
        
        # Labels stringently fill NaN with 0.0
        self.y = y_df.fillna(0.0).values.astype(np.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]), 
            torch.tensor(self.y[idx]), 
            torch.tensor(self.mask[idx])
        )

def create_kfold_splits(features_df, labels_df, n_folds=5, seed=42):
    np.random.seed(seed)
    disease_labels = ["obesity", "t2d", "hypertension", "cad", "ckd", "stroke", "osteoporosis"]
    
    # We substitute NaNs with a dummy class 2 for stratification
    y_strat = labels_df[disease_labels].fillna(2).values.astype(int)
    X_dummy = np.zeros((len(labels_df), 1))
    
    stratifier = IterativeStratification(n_splits=n_folds, order=1)
        
    splits = []
    for train_idx, val_idx in stratifier.split(X_dummy, y_strat):
        splits.append((train_idx, val_idx))
        
    return splits

def get_dataloaders(fold_idx, features_df, labels_df, feature_names, batch_size=256, seed=42):
    splits = create_kfold_splits(features_df, labels_df, n_folds=5, seed=seed)
    train_idx, val_idx = splits[fold_idx]
    
    train_feat = features_df.iloc[train_idx].copy().reset_index(drop=True)
    val_feat = features_df.iloc[val_idx].copy().reset_index(drop=True)
    train_labels = labels_df.iloc[train_idx].copy().reset_index(drop=True)
    val_labels = labels_df.iloc[val_idx].copy().reset_index(drop=True)
    
    # Isolate continuous variables (>2 unique values)
    cont_cols = train_feat.select_dtypes(include=[np.number]).nunique()
    cont_cols = cont_cols[cont_cols > 2].index.tolist()
    if "SEQN" in cont_cols:
        cont_cols.remove("SEQN")
        
    # Fit QuantileTransformer on train fold only
    if len(cont_cols) > 0:
        qt = QuantileTransformer(output_distribution='normal', random_state=seed)
        train_feat[cont_cols] = qt.fit_transform(train_feat[cont_cols])
        val_feat[cont_cols] = qt.transform(val_feat[cont_cols])
    
    train_ds = ComorbidityDataset(train_feat, train_labels, feature_names)
    val_ds = ComorbidityDataset(val_feat, val_labels, feature_names)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    import json
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if base == "": base = "."
    df_f = pd.read_csv(os.path.join(base, "data/final/features.csv"))
    df_l = pd.read_csv(os.path.join(base, "data/final/labels.csv"))
    with open(os.path.join(base, "data/final/feature_names.json")) as f:
        feature_names = json.load(f)
        
    print("Testing create_kfold_splits...")
    splits = create_kfold_splits(df_f, df_l)
    print(f"Number of tuples returned: {len(splits)}")
    
    t0, v0 = splits[0]
    t1, v1 = splits[1]
    ob0 = df_l.iloc[v0]['obesity'].mean() * 100
    ob1 = df_l.iloc[v1]['obesity'].mean() * 100
    
    print(f"Obesity prevalence in fold 0 val: {ob0:.2f}%")
    print(f"Obesity prevalence in fold 1 val: {ob1:.2f}%")
    
    print("\nTesting get_dataloaders and __getitem__ shapes...")
    train_loader, val_loader = get_dataloaders(0, df_f, df_l, feature_names)
    x, y, mask = next(iter(train_loader))
    
    print("First batch shapes:")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"mask shape: {mask.shape}")
    
    print("\nVerifying mask tensor logic:")
    print(f"y[0]: {y[0]}")
    print(f"mask[0]: {mask[0]}")
