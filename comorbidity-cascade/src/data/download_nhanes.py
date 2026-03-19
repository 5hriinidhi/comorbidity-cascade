import os
import yaml
import time
import pyreadstat
import pandas as pd
from pathlib import Path
import urllib.request
import urllib.error

SUFFIX_MAP = {
    "2015-2016": "I",
    "2017-2018": "J",
    "2019-2020": "L"
}

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_file(url, out_path, retries=3, backoff=5):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response, open(out_path, 'wb') as out_file:
                # Read first chunk to verify it's not the 404 HTML
                data = response.read(1024)
                if b"Page Not Found" in data or b"<html" in data.lower()[:50]:
                    return False
                out_file.write(data)
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    out_file.write(chunk)
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(backoff)
    return False

def main():
    config = load_config()
    raw_data_dir = config["paths"]["raw_data"]
    cycles = config["nhanes"]["cycles"]
    modules = config["nhanes"]["modules"]
    
    Path(raw_data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"{'MODULE':<10} | {'CYCLE':<11} | {'ROWS':<6} | {'COLS':<4} | {'STATUS':<15}")
    print("-" * 55)
    
    for cycle in cycles:
        suffix = SUFFIX_MAP.get(cycle)
        start_year = cycle.split('-')[0]
        cycle_dir = os.path.join(raw_data_dir, cycle)
        Path(cycle_dir).mkdir(parents=True, exist_ok=True)
        
        for key, module in modules.items():
            url_modern = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start_year}/DataFiles/{module}_{suffix}.xpt"
            url_legacy = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{module}_{suffix}.XPT"
            url_pandemic = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_{module}.xpt"
            
            xpt_path = os.path.join(cycle_dir, f"{module}_{suffix}.XPT")
            csv_path = os.path.join(cycle_dir, f"{module}_{suffix}.csv")
            
            # Cleanup broken HTML
            if os.path.exists(xpt_path) and os.path.getsize(xpt_path) < 50000:
                os.remove(xpt_path)
            
            # Download
            if not os.path.exists(xpt_path) or os.path.getsize(xpt_path) == 0:
                success = download_file(url_modern, xpt_path)
                if not success:
                    success = download_file(url_legacy, xpt_path)
                if not success and cycle == "2019-2020":
                    success = download_file(url_pandemic, xpt_path)
                    
                if not success:
                    print(f"{module:<10} | {cycle:<11} | {'-':<6} | {'-':<4} | {'FAILED (404/Net)':<15}")
                    continue
            
            # Parse and Verify
            try:
                if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                    df, meta = pyreadstat.read_xport(xpt_path)
                    df.columns = [str(c).upper() for c in df.columns]
                    df.to_csv(csv_path, index=False)
                else:
                    df = pd.read_csv(csv_path, low_memory=False)
                    
                rows, cols = df.shape
                status = "OK"
                
                if module == "DEMO" and "SEQN" not in df.columns:
                    status = "FAILED (NO SEQN)"
                elif module == "DIQ" and "DIQ010" not in df.columns:
                    status = "FAILED (NO DIQ)"
                elif module == "BPQ" and "BPQ020" not in df.columns:
                    status = "FAILED (NO BPQ)"
                    
                print(f"{module:<10} | {cycle:<11} | {rows:<6} | {cols:<4} | {status:<15}")
                
            except Exception as e:
                print(f"{module:<10} | {cycle:<11} | {'-':<6} | {'-':<4} | FAILED (Parse {str(e)[:5]})")

if __name__ == "__main__":
    main()
