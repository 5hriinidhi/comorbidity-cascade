import os
import pandas as pd
from pathlib import Path

print("--- FINAL CHECKPOINT VERIFICATION ---")

base = '.'

# Subdirectories
cycles = ["2015-2016", "2017-2018", "2019-2020"]
for c in cycles:
    p = os.path.join(base, "data/raw", c)
    if os.path.exists(p):
        xpt = list(Path(p).glob("*.XPT"))
        csv = list(Path(p).glob("*.csv"))
        print(f"Cycle {c}: {len(xpt)} XPT, {len(csv)} CSV")
        
        # Spot checks
        demo = os.path.join(p, f"DEMO_{'I' if c=='2015-2016' else 'J' if c=='2017-2018' else 'L'}.csv")
        if os.path.exists(demo):
            df_demo = pd.read_csv(demo, nrows=5)
            print(f"  {c} DEMO size: {os.path.getsize(demo)/1024/1024:.2f} MB, SEQN exists: {'SEQN' in df_demo.columns}")
            
        diq = os.path.join(p, f"DIQ_{'I' if c=='2015-2016' else 'J' if c=='2017-2018' else 'L'}.csv")
        if os.path.exists(diq):
            df_diq = pd.read_csv(diq, nrows=5)
            print(f"  {c} DIQ010 exists: {'DIQ010' in df_diq.columns}")
            
        bpq = os.path.join(p, f"BPQ_{'I' if c=='2015-2016' else 'J' if c=='2017-2018' else 'L'}.csv")
        if os.path.exists(bpq):
            df_bpq = pd.read_csv(bpq, nrows=5)
            print(f"  {c} BPQ020 exists: {'BPQ020' in df_bpq.columns}")

merged_file = os.path.join(base, "data/processed/nhanes_merged.csv")
if os.path.exists(merged_file):
    print(f"\nnhanes_merged.csv exists. Size: {os.path.getsize(merged_file)/1024/1024:.2f} MB")
    df = pd.read_csv(merged_file, low_memory=False)
    print(f"Row count: {len(df)} (>= 10,000?: {len(df) >= 10000})")
    print(f"SEQN unique?: {df['SEQN'].is_unique}")
    
    if "RIDAGEYR" in df.columns:
        print(f"RIDAGEYR min: {df['RIDAGEYR'].min()}, max: {df['RIDAGEYR'].max()}")
        
    print(f"cycle_id values: {list(df['cycle_id'].unique())}")
    
    print("\nTarget Variables with >40% missing:")
    targets = ["SEQN", "RIDAGEYR", "RIAGENDR", "DIQ010", "BPQ020", "MCQ160C", "KIQ022", "BMXBMI", "BPXSY1", "BPXDI1", "LBXGLU", "LBXGH", "LBXTC", "LBDLDL", "LBDHDD", "LBXTR", "LBXSCR", "PAD680", "SLD010H", "SMQ020"]
    for col in targets:
        if col in df.columns:
            pct = df[col].isna().mean()
            if pct > 0.4:
                print(f"  {col}: {pct*100:.1f}%")
        else:
            print(f"  {col}: NOT FOUND (100% missing)")
else:
    print("nhanes_merged.csv NOT FOUND!")
