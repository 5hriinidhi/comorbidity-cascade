import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_performance_plot():
    # Load all results
    results_dir = "results"
    baseline_df = pd.read_csv(os.path.join(results_dir, "baseline_auroc.csv"))
    flat_df = pd.read_csv(os.path.join(results_dir, "mtl_flat_auroc.csv"))
    graph_df = pd.read_csv(os.path.join(results_dir, "mtl_graph_auroc.csv"))
    full_df = pd.read_csv(os.path.join(results_dir, "mtl_full_auroc.csv"))
    
    baseline = baseline_df.groupby("disease")["auroc"].mean().to_dict()
    flat = flat_df.groupby("disease")["auroc"].mean().to_dict()
    graph = graph_df.groupby("disease")["auroc"].mean().to_dict()
    full = full_df.groupby("disease")["auroc"].mean().to_dict()
    
    # Merge into long format for plotting
    data = []
    diseases = sorted(list(baseline.keys()))
    for d in diseases:
        data.append({"Disease": d.capitalize(), "Model": "CatBoost (Baseline)", "AUROC": baseline[d]})
        data.append({"Disease": d.capitalize(), "Model": "MTL_Flat", "AUROC": flat[d]})
        data.append({"Disease": d.capitalize(), "Model": "MTL_Graph", "AUROC": graph[d]})
        data.append({"Disease": d.capitalize(), "Model": "MTL_Full (Proposed)", "AUROC": full[d]})
    
    df = pd.DataFrame(data)
    
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Use a premium color palette
    palette = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]
    
    ax = sns.barplot(data=df, x="Disease", y="AUROC", hue="Model", palette=palette)
    
    # Add titles and labels
    plt.title("Comorbidity Cascade: Architecture Progression Model Performance", fontsize=18, pad=20, weight='bold')
    plt.ylabel("Mean AUROC (5-Fold CV)", fontsize=14)
    plt.xlabel("Target Disease", fontsize=14)
    plt.ylim(0.4, 1.05)
    plt.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate significant gains (Stroke)
    stroke_idx = diseases.index('stroke')
    stroke_full = full['stroke']
    stroke_cat = baseline['stroke']
    ax.annotate(f"GAINS: +{stroke_full-stroke_cat:.3f}", 
                xy=(stroke_idx, stroke_full), xytext=(stroke_idx, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=12, color='darkgreen', weight='bold', ha='center')

    plt.tight_layout()
    plt.savefig("results/model_comparison_plot.png", dpi=300)
    print("Plot saved to results/model_comparison_plot.png")

if __name__ == "__main__":
    generate_performance_plot()
