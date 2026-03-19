import os
import json
import pandas as pd

print("--- PHASE 3 CHECKPOINT VERIFICATION ---")
base = '.' # running from comorbidity-cascade
feat_path = os.path.join(base, "data/final/features.csv")
lab_path = os.path.join(base, "data/final/labels.csv")
names_path = os.path.join(base, "data/final/feature_names.json")

print(f"features.csv exists: {os.path.exists(feat_path)}")
print(f"labels.csv exists: {os.path.exists(lab_path)}")
print(f"feature_names.json exists: {os.path.exists(names_path)}")

if os.path.exists(feat_path) and os.path.exists(lab_path):
    df_f = pd.read_csv(feat_path)
    df_l = pd.read_csv(lab_path)
    
    print(f"Row match: {len(df_f) == len(df_l)} ({len(df_f)} rows)")
    
    with open(names_path, "r") as f:
        names = json.load(f)
    print(f"feature_names.json length: {len(names)} (>= 25?: {len(names) >= 25})")
    
    prs_cols = [c for c in df_f.columns if c.startswith("prs_") and c != "prs_available"]
    prs_zeros = all((df_f[c] == 0).all() for c in prs_cols)
    print(f"PRS columns ({len(prs_cols)}) all zeros: {prs_zeros}")
    
    print(f"prs_available all zeros: {(df_f['prs_available'] == 0).all()}")
    
    print(f"No NaN in features.csv: {not df_f.isna().any().any()}")
    
    print(f"No NaN in obesity label: {not df_l['obesity'].isna().any()}")
    
    print(f"Feature Breakdown: Biometric+Lifestyle={len(names)-8}, PRS=8, Total={len(names)}")
