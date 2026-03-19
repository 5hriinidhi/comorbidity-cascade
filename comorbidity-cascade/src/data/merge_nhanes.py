import os
import yaml
import pandas as pd
from pathlib import Path

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    raw_dir = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_data"]
    cycles = config["nhanes"]["cycles"]
    modules = config["nhanes"]["modules"]
    
    SUFFIX_MAP = {
        "2015-2016": "I",
        "2017-2018": "J",
        "2019-2020": "L"
    }

    cycle_dfs = []
    
    for i, cycle in enumerate(cycles):
        suffix = SUFFIX_MAP.get(cycle)
        cycle_dir = os.path.join(raw_dir, cycle)
        
        demo_mod = "DEMO"
        demo_path = os.path.join(cycle_dir, f"{demo_mod}_{suffix}.csv")
        
        if not os.path.exists(demo_path):
            print(f"Warning: {demo_path} not found. Skipping cycle.")
            continue
            
        merged_df = pd.read_csv(demo_path, low_memory=False)
        merged_df["cycle_id"] = i
        
        for key, module in modules.items():
            if module == "DEMO":
                continue
                
            csv_path = os.path.join(cycle_dir, f"{module}_{suffix}.csv")
            if os.path.exists(csv_path):
                mod_df = pd.read_csv(csv_path, low_memory=False)
                if "SEQN" in mod_df.columns:
                    merged_df = pd.merge(merged_df, mod_df, on="SEQN", how="left", suffixes=("", f"_{module}"))
                else:
                    print(f"Warning: SEQN not in {csv_path}")
            else:
                print(f"Warning: {csv_path} not found.")
                
        cycle_dfs.append(merged_df)
        
    if not cycle_dfs:
        print("No data found to merge.")
        return
        
    master_df = pd.concat(cycle_dfs, ignore_index=True)
    rows_before = len(master_df)
    
    if "RIDAGEYR" in master_df.columns:
        master_df = master_df[master_df["RIDAGEYR"] >= 18]
    rows_after_adult = len(master_df)
    
    disease_cols = ["DIQ010", "BPQ020", "MCQ160C", "KIQ022"]
    for col in disease_cols:
        if col not in master_df.columns:
            master_df[col] = pd.NA
            
    master_df = master_df.dropna(subset=disease_cols, how="all")
    rows_after_missing = len(master_df)
    cols_count = len(master_df.columns)
    
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(processed_dir, "nhanes_merged.csv")
    master_df.to_csv(out_file, index=False)
    
    print("--- MERGE SUMMARY ---")
    print(f"Total rows before adult filter:      {rows_before}")
    print(f"Total rows after adult filter:       {rows_after_adult}")
    print(f"Total rows after missingness filter: {rows_after_missing}")
    print(f"Total columns:                       {cols_count}")
    
    print("\nMissing value percentage per key variable:")
    key_vars = [
        "SEQN", "RIDAGEYR", "RIAGENDR", "DIQ010", "BPQ020", "MCQ160C", "KIQ022",
        "BMXBMI", "BPXSY1", "BPXDI1", "LBXGLU", "LBXGH", "LBXTC", "LBDLDL", "LBDHDD",
        "LBXTR", "LBXSCR", "PAD680", "SLD010H", "SMQ020"
    ]
    
    print(f"{'VARIABLE':<15} | {'MISSING %':<10}")
    print("-" * 30)
    for kv in key_vars:
        if kv in master_df.columns:
            pct = master_df[kv].isna().mean() * 100
            print(f"{kv:<15} | {pct:.2f}%")
        else:
            print(f"{kv:<15} | NOT FOUND")

if __name__ == "__main__":
    main()
