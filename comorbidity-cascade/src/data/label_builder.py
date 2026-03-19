import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_binary(col_name, df):
    if col_name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = pd.to_numeric(df[col_name], errors='coerce')
    # 7, 9, 77, 99 are refused/don't know in NHANES
    s = s.replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
    out = pd.Series(np.nan, index=s.index)
    out[s == 1] = 1
    out[(s.notna()) & (s != 1)] = 0
    return out

def main():
    config = load_config()
    processed_dir = config["paths"]["processed_data"]
    infile = os.path.join(processed_dir, "nhanes_merged.csv")
    outfile = os.path.join(processed_dir, "nhanes_labels.csv")
    
    if not os.path.exists(infile):
        print(f"Error: {infile} not found. Run merge script first.")
        return
        
    df = pd.read_csv(infile, low_memory=False)
    labels_df = pd.DataFrame()
    labels_df["SEQN"] = df["SEQN"]
    
    # 1. Obesity
    if "BMXBMI" in df.columns:
        bmx = pd.to_numeric(df["BMXBMI"], errors='coerce')
        labels_df["obesity"] = (bmx >= 30.0).astype(float)
        labels_df.loc[bmx.isna(), "obesity"] = np.nan
    else:
        labels_df["obesity"] = np.nan
        
    # 2. T2D
    labels_df["t2d"] = get_binary("DIQ010", df)
    
    # 3. Hypertension
    labels_df["hypertension"] = get_binary("BPQ020", df)
    
    # 4. CAD
    mcq = get_binary("MCQ160C", df)
    cdq = get_binary("CDQ008", df)
    cad = pd.Series(np.nan, index=df.index)
    cad[(mcq == 1) | (cdq == 1)] = 1
    cad[(mcq == 0) & (cdq != 1)] = 0
    cad[(cdq == 0) & (mcq != 1)] = 0
    labels_df["cad"] = cad
    
    # 5. CKD
    labels_df["ckd"] = get_binary("KIQ022", df)
    
    # 6. Stroke
    labels_df["stroke"] = get_binary("MCQ160F", df)
    
    # 7. Osteoporosis
    labels_df["osteoporosis"] = get_binary("OSQ060", df)
    
    # Drop rows where obesity is NaN
    labels_df = labels_df.dropna(subset=["obesity"])
    
    # Only keep the output columns
    out_cols = ["SEQN", "obesity", "t2d", "hypertension", "cad", "ckd", "stroke", "osteoporosis"]
    labels_df = labels_df[out_cols]
    
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(outfile, index=False)
    
    print("--- LABEL PREVALENCE ---")
    print(f"{'disease':<15} | {'n_positive':<10} | {'n_total':<10} | {'prevalence_pct':<15}")
    print("-" * 55)
    for col in out_cols[1:]:
        n_pos = int(labels_df[col].sum()) if not labels_df[col].isna().all() else 0
        n_tot = int(labels_df[col].notna().sum())
        prev = (n_pos / n_tot * 100) if n_tot > 0 else 0
        print(f"{col:<15} | {n_pos:<10} | {n_tot:<10} | {prev:.2f}%")

if __name__ == "__main__":
    main()
