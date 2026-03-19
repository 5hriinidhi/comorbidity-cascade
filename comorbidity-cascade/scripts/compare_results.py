import pandas as pd
import numpy as np
import os

def load_results(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    # df has columns: fold, disease, auroc
    return df.groupby('disease')['auroc'].mean().to_dict()

def main():
    base_dir = "."
    catboost_file = os.path.join(base_dir, "results/baseline_auroc.csv")
    mtl_flat_file = os.path.join(base_dir, "results/mtl_flat_auroc.csv")
    mtl_graph_file = os.path.join(base_dir, "results/mtl_graph_auroc.csv")
    
    catboost_res = load_results(catboost_file) or {}
    mtl_flat_res = load_results(mtl_flat_file) or {}
    mtl_graph_res = load_results(mtl_graph_file) or {}
    
    # Diseases order from config
    diseases = ["obesity", "t2d", "hypertension", "cad", "ckd", "stroke", "osteoporosis"]
    
    print(f"{'Disease':<15} | {'CatBoost':<10} | {'MTL_Flat':<10} | {'MTL_Graph':<10}")
    print("-" * 55)
    
    macro_catboost = []
    macro_mtl_flat = []
    macro_mtl_graph = []
    
    for d in diseases:
        cb = catboost_res.get(d, np.nan)
        mf = mtl_flat_res.get(d, np.nan)
        mg = mtl_graph_res.get(d, np.nan)
        
        if not np.isnan(cb): macro_catboost.append(cb)
        if not np.isnan(mf): macro_mtl_flat.append(mf)
        if not np.isnan(mg): macro_mtl_graph.append(mg)
        
        cb_str = f"{cb:.4f}" if not np.isnan(cb) else "N/A"
        mf_str = f"{mf:.4f}" if not np.isnan(mf) else "N/A"
        mg_str = f"{mg:.4f}" if not np.isnan(mg) else "N/A"
        
        highlight = ""
        if not np.isnan(cb) and not np.isnan(mg):
            if mg - cb > 0.02:
                highlight = ">>>"
                
        print(f"{d:<15} | {cb_str:<10} | {mf_str:<10} | {mg_str:<10} {highlight}")
        
    print("-" * 55)
    m_cb = np.mean(macro_catboost) if macro_catboost else np.nan
    m_mf = np.mean(macro_mtl_flat) if macro_mtl_flat else np.nan
    m_mg = np.mean(macro_mtl_graph) if macro_mtl_graph else np.nan
    
    m_cb_str = f"{m_cb:.4f}" if not np.isnan(m_cb) else "N/A"
    m_mf_str = f"{m_mf:.4f}" if not np.isnan(m_mf) else "N/A"
    m_mg_str = f"{m_mg:.4f}" if not np.isnan(m_mg) else "N/A"
    
    print(f"{'Macro-AUROC':<15} | {m_cb_str:<10} | {m_mf_str:<10} | {m_mg_str:<10}")

if __name__ == "__main__":
    main()
