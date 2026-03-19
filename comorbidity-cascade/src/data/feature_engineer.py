import os
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    processed_dir = config["paths"]["processed_data"]
    final_dir = config["paths"]["final_data"]
    
    merged_path = os.path.join(processed_dir, "nhanes_merged.csv")
    labels_path = os.path.join(processed_dir, "nhanes_labels.csv")
    
    if not os.path.exists(merged_path) or not os.path.exists(labels_path):
        print("Error: Required CSV files not found.")
        return
        
    df_merged = pd.read_csv(merged_path, low_memory=False)
    df_labels = pd.read_csv(labels_path)
    
    df = pd.merge(df_merged, df_labels, on="SEQN", how="inner")
    
    # 1. Continuous features
    cont_cols = [
        "BPXSY1", "BPXDI1", "LBXGLU", "LBXGH", "BMXBMI", "BMXWAIST", 
        "LBXTC", "LBDLDL", "LBDHDD", "LBXTR", "LBXSCR", 
        "DR1TKCAL", "DR1TFIBE", "DR1TSFAT", "PAD680", "SLD010H", "SMD030"
    ]
    
    for c in cont_cols:
        if c not in df.columns:
            df[c] = np.nan
            
    # Handle totally missing columns for SimpleImputer compatibility
    for c in cont_cols:
        if df[c].isna().all():
            df[c] = 0.0
            
    print("WARNING: PRS not computed — placeholder zeros used. Replace with PRSice-2 output if genotype data becomes available.")
    disease_labels = config["diseases"]["labels"]
    prs_cols = [f"prs_{d}" for d in disease_labels]
    for c in prs_cols:
        df[c] = 0.0
    df["prs_available"] = 0.0
    
    # 2. Categorical features
    if "SMQ020" not in df.columns:
        df["SMQ020"] = np.nan
    smq = pd.to_numeric(df["SMQ020"], errors='coerce').replace({7: np.nan, 9: np.nan})
    smq = smq.map({1: "Ever", 2: "Never"}).fillna("Unknown")
    
    if "ALQ130" not in df.columns:
        df["ALQ130"] = np.nan
    alq = pd.to_numeric(df["ALQ130"], errors='coerce').replace({777: np.nan, 999: np.nan})
    alq_binned = pd.cut(alq, bins=[-1, 0.5, 7.5, 14.5, 1000], labels=["0", "1-7", "8-14", "15+"])
    alq_binned = alq_binned.astype(str).replace('nan', 'Unknown')
    
    df_smq_oh = pd.get_dummies(smq, prefix="SMQ020").astype(float)
    df_alq_oh = pd.get_dummies(alq_binned, prefix="ALQ130").astype(float)
    cat_cols = list(df_smq_oh.columns) + list(df_alq_oh.columns)
    
    # 3. Impute & Scale continuous
    X_cont = df[cont_cols].copy()
    
    imputer = SimpleImputer(strategy="median")
    X_cont_imp = imputer.fit_transform(X_cont)
    
    scaler = QuantileTransformer(output_distribution="normal")
    X_cont_scaled = scaler.fit_transform(X_cont_imp)
    
    df_cont = pd.DataFrame(X_cont_scaled, columns=cont_cols, index=df.index)
    
    # Combine features
    prs_df = df[prs_cols + ["prs_available"]].copy()
    df_features = pd.concat([df[["SEQN"]], df_cont, df_smq_oh, df_alq_oh, prs_df], axis=1)
    
    df_out_labels = df[["SEQN"] + disease_labels].copy()
    
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    df_features.to_csv(os.path.join(final_dir, "features.csv"), index=False)
    df_out_labels.to_csv(os.path.join(final_dir, "labels.csv"), index=False)
    
    feature_names = list(df_features.columns)
    feature_names.remove("SEQN")
    with open(os.path.join(final_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=4)
        
    print(f"Features shape: {df_features.shape}")
    print(f"Labels shape:   {df_out_labels.shape}")
    print(f"Feature count breakdown:")
    print(f"  Continuous:  {len(cont_cols)}")
    print(f"  Categorical: {len(cat_cols)}")
    print(f"  PRS Proxies: {len(prs_cols) + 1}")
    print(f"  Total Features (excl SEQN): {len(feature_names)}")

if __name__ == "__main__":
    main()
