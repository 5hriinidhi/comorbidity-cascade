import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_extra_plots():
    results_dir = "results"
    
    # 1. Causal Consistency Trend
    hparam_df = pd.read_csv(os.path.join(results_dir, "hparam_search_log.csv"))
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="white")
    
    # Dual axis plot: AUROC and Inversion Rate
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Lambda (Causal Weight)', fontsize=12)
    ax1.set_ylabel('Val Macro-AUROC', color=color, fontsize=12)
    ax1.plot(hparam_df['lambda'], hparam_df['val_macro_auroc'], marker='o', color=color, linewidth=2, label='AUROC')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Causal Inversion Rate (%)', color=color, fontsize=12)
    ax2.plot(hparam_df['lambda'], hparam_df['causal_inversion_rate_%'], marker='s', color=color, linestyle='--', linewidth=2, label='Inversion Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Model Trade-off: Performance vs. Causal Consistency", fontsize=14, weight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "causal_tradeoff.png"), dpi=300)
    
    # 2. Intervention Heatmap
    interv_df = pd.read_csv(os.path.join(results_dir, "intervention_deltas.csv"))
    # Pivot for heatmap
    pivot_df = interv_df.pivot(index="intervention", columns="disease", values="delta_pct")
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Risk Delta %'})
    plt.title("Intervention Impact Heatmap: Lifestyle Factors vs. Disease Risk", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Target Disease")
    plt.ylabel("Intervention Type")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "intervention_heatmap.png"), dpi=300)
    
    print("Extra plots saved to results/")

if __name__ == "__main__":
    generate_extra_plots()
