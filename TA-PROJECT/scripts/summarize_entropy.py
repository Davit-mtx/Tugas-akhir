import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CSV = PROJECT_ROOT / "data/results/baseline/summary_all_results.csv"
OPTIMIZED_CSV = PROJECT_ROOT / "data/results/optimized/summary_optimized_results.csv"
OUT_DIR = PROJECT_ROOT / "data/results/plots"

def main():
    print("=== PROSES SUMMARY NILAI ENTROPY (BASELINE VS OPTIMIZED) ===")
    
    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if input CSV files exist
    if not BASELINE_CSV.exists():
        print(f"[ERROR] File baseline tidak ditemukan di: {BASELINE_CSV}")
        return
    if not OPTIMIZED_CSV.exists():
        print(f"[ERROR] File optimized tidak ditemukan di: {OPTIMIZED_CSV}")
        return
        
    print(f"Loading baseline from: {BASELINE_CSV.name}")
    df_base = pd.read_csv(BASELINE_CSV)
    
    print(f"Loading optimized from: {OPTIMIZED_CSV.name}")
    df_opt = pd.read_csv(OPTIMIZED_CSV)
    
    # Required columns
    req_cols = ["File Name", "Category", "Entropy_R", "Entropy_G", "Entropy_B", "Entropy"]
    
    # Verify columns exist
    for col in req_cols:
        if col not in df_base.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di file baseline.")
        if col not in df_opt.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di file optimized.")
            
    # Filter only relevant columns
    df_base_filtered = df_base[req_cols].copy()
    df_opt_filtered = df_opt[req_cols].copy()
    
    # Rename columns for baseline and optimized before merge
    df_base_filtered = df_base_filtered.rename(columns={
        "Entropy_R": "Entropy_R_baseline",
        "Entropy_G": "Entropy_G_baseline",
        "Entropy_B": "Entropy_B_baseline",
        "Entropy": "entropi_baseline"
    })
    
    df_opt_filtered = df_opt_filtered.rename(columns={
        "Entropy_R": "Entropy_R_optimized",
        "Entropy_G": "Entropy_G_optimized",
        "Entropy_B": "Entropy_B_optimized",
        "Entropy": "entropi_optimized"
    })
    
    # Merge datasets on File Name and Category
    merged_df = pd.merge(df_base_filtered, df_opt_filtered, on=["File Name", "Category"])
    
    # Calculate difference (selisih) of overall average entropy (entropi_optimized - entropi_baseline)
    merged_df["selisih"] = merged_df["entropi_optimized"] - merged_df["entropi_baseline"]
    
    # Reorder columns for readability
    col_order = [
        "File Name", "Category",
        "Entropy_R_baseline", "Entropy_G_baseline", "Entropy_B_baseline", "entropi_baseline",
        "Entropy_R_optimized", "Entropy_G_optimized", "Entropy_B_optimized", "entropi_optimized",
        "selisih"
    ]
    merged_df = merged_df[col_order]
    
    # Save to CSV in results/plots/
    out_csv_path = OUT_DIR / "summary_entropy.csv"
    merged_df.to_csv(out_csv_path, index=False)
    print(f"[SUKSES] File summary CSV disimpan di: {out_csv_path}")
    
    # Generate visual comparison plot (Boxplot of entropy and bar plot of averages)
    # Set plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Boxplot comparing baseline and optimized entropy values by category
    melted_df = pd.melt(
        merged_df, 
        id_vars=["Category"], 
        value_vars=["entropi_baseline", "entropi_optimized"],
        var_name="Metode", 
        value_name="Entropy"
    )
    # Rename methods for plotting
    melted_df["Metode"] = melted_df["Metode"].map({
        "entropi_baseline": "Baseline",
        "entropi_optimized": "Optimized"
    })
    
    sns.boxplot(
        ax=axes[0],
        data=melted_df,
        x="Category",
        y="Entropy",
        hue="Metode",
        palette="Set2"
    )
    axes[0].set_title("Distribusi Nilai Entropy: Baseline vs Optimized")
    axes[0].set_xlabel("Kategori Kualitas Image")
    axes[0].set_ylabel("Nilai Entropy")
    
    # Right subplot: Barplot of average differences per category
    category_diffs = merged_df.groupby("Category")["selisih"].mean().reset_index()
    sns.barplot(
        ax=axes[1],
        data=category_diffs,
        x="Category",
        y="selisih",
        hue="Category",
        legend=False,
        palette="coolwarm"
    )
    axes[1].set_title("Rata-rata Selisih Entropy per Kategori\n(Optimized - Baseline)")
    axes[1].set_xlabel("Kategori Kualitas Image")
    axes[1].set_ylabel("Rata-rata Selisih Entropy")
    
    # Add values on top of the bars
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(
            f"{height:+.6f}",
            xy=(p.get_x() + p.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        
    plt.tight_layout()
    out_plot_path = OUT_DIR / "summary_entropy_comparison_plot.png"
    plt.savefig(out_plot_path, dpi=300)
    plt.close()
    print(f"[SUKSES] Grafik perbandingan disimpan di: {out_plot_path}")

if __name__ == "__main__":
    main()
