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
    print("=== PROSES SUMMARY NILAI KORELASI (BASELINE VS OPTIMIZED) ===")
    
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
    
    # Core columns we need to extract
    req_cols = [
        "File Name", "Category",
        "Corr_Horizontal_R", "Corr_Vertical_R", "Corr_Diagonal_R",
        "Corr_Horizontal_G", "Corr_Vertical_G", "Corr_Diagonal_G",
        "Corr_Horizontal_B", "Corr_Vertical_B", "Corr_Diagonal_B",
        "Corr_Horizontal", "Corr_Vertical", "Corr_Diagonal"
    ]
    
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
    base_rename = {
        "Corr_Horizontal_R": "Corr_Horizontal_R_baseline",
        "Corr_Vertical_R": "Corr_Vertical_R_baseline",
        "Corr_Diagonal_R": "Corr_Diagonal_R_baseline",
        "Corr_Horizontal_G": "Corr_Horizontal_G_baseline",
        "Corr_Vertical_G": "Corr_Vertical_G_baseline",
        "Corr_Diagonal_G": "Corr_Diagonal_G_baseline",
        "Corr_Horizontal_B": "Corr_Horizontal_B_baseline",
        "Corr_Vertical_B": "Corr_Vertical_B_baseline",
        "Corr_Diagonal_B": "Corr_Diagonal_B_baseline",
        "Corr_Horizontal": "Corr_Horizontal_baseline",
        "Corr_Vertical": "Corr_Vertical_baseline",
        "Corr_Diagonal": "Corr_Diagonal_baseline"
    }
    
    opt_rename = {
        "Corr_Horizontal_R": "Corr_Horizontal_R_optimized",
        "Corr_Vertical_R": "Corr_Vertical_R_optimized",
        "Corr_Diagonal_R": "Corr_Diagonal_R_optimized",
        "Corr_Horizontal_G": "Corr_Horizontal_G_optimized",
        "Corr_Vertical_G": "Corr_Vertical_G_optimized",
        "Corr_Diagonal_G": "Corr_Diagonal_G_optimized",
        "Corr_Horizontal_B": "Corr_Horizontal_B_optimized",
        "Corr_Vertical_B": "Corr_Vertical_B_optimized",
        "Corr_Diagonal_B": "Corr_Diagonal_B_optimized",
        "Corr_Horizontal": "Corr_Horizontal_optimized",
        "Corr_Vertical": "Corr_Vertical_optimized",
        "Corr_Diagonal": "Corr_Diagonal_optimized"
    }
    
    df_base_filtered = df_base_filtered.rename(columns=base_rename)
    df_opt_filtered = df_opt_filtered.rename(columns=opt_rename)
    
    # Merge datasets on File Name and Category
    merged_df = pd.merge(df_base_filtered, df_opt_filtered, on=["File Name", "Category"])
    
    # Calculate absolute differences (selisih = |optimized - baseline|) for korelasi rata-rata
    merged_df["selisih_horizontal"] = (merged_df["Corr_Horizontal_optimized"] - merged_df["Corr_Horizontal_baseline"]).abs()
    merged_df["selisih_vertical"] = (merged_df["Corr_Vertical_optimized"] - merged_df["Corr_Vertical_baseline"]).abs()
    merged_df["selisih_diagonal"] = (merged_df["Corr_Diagonal_optimized"] - merged_df["Corr_Diagonal_baseline"]).abs()
    
    # Reorder columns logically
    col_order = [
        "File Name", "Category",
        
        # Horizontal
        "Corr_Horizontal_R_baseline", "Corr_Horizontal_G_baseline", "Corr_Horizontal_B_baseline", "Corr_Horizontal_baseline",
        "Corr_Horizontal_R_optimized", "Corr_Horizontal_G_optimized", "Corr_Horizontal_B_optimized", "Corr_Horizontal_optimized",
        "selisih_horizontal",
        
        # Vertical
        "Corr_Vertical_R_baseline", "Corr_Vertical_G_baseline", "Corr_Vertical_B_baseline", "Corr_Vertical_baseline",
        "Corr_Vertical_R_optimized", "Corr_Vertical_G_optimized", "Corr_Vertical_B_optimized", "Corr_Vertical_optimized",
        "selisih_vertical",
        
        # Diagonal
        "Corr_Diagonal_R_baseline", "Corr_Diagonal_G_baseline", "Corr_Diagonal_B_baseline", "Corr_Diagonal_baseline",
        "Corr_Diagonal_R_optimized", "Corr_Diagonal_G_optimized", "Corr_Diagonal_B_optimized", "Corr_Diagonal_optimized",
        "selisih_diagonal"
    ]
    merged_df = merged_df[col_order]
    
    # Save to CSV in results/plots/
    out_csv_path = OUT_DIR / "summary_correlation.csv"
    merged_df.to_csv(out_csv_path, index=False)
    print(f"[SUKSES] File summary korelasi CSV disimpan di: {out_csv_path}")
    
    # Generate visual comparison plot (4 subplots in 2x2 grid)
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    directions = ["Horizontal", "Vertical", "Diagonal"]
    
    for i, direction in enumerate(directions):
        ax = axes[i // 2, i % 2]
        
        # Melt data for current direction
        base_col = f"Corr_{direction}_baseline"
        opt_col = f"Corr_{direction}_optimized"
        
        melted = pd.melt(
            merged_df,
            id_vars=["Category"],
            value_vars=[base_col, opt_col],
            var_name="Metode",
            value_name="Correlation"
        )
        melted["Metode"] = melted["Metode"].map({
            base_col: "Baseline",
            opt_col: "Optimized"
        })
        
        sns.boxplot(
            ax=ax,
            data=melted,
            x="Category",
            y="Correlation",
            hue="Metode",
            palette="Set2"
        )
        ax.set_title(f"Distribusi Korelasi {direction}: Baseline vs Optimized")
        ax.set_xlabel("Kategori Kualitas Image")
        ax.set_ylabel(f"Koefisien Korelasi {direction}")
        
    # 4th subplot: Barplot of average absolute differences (selisih) per category
    ax_bar = axes[1, 1]
    
    # Average of differences grouped by Category
    avg_diffs = merged_df.groupby("Category")[["selisih_horizontal", "selisih_vertical", "selisih_diagonal"]].mean().reset_index()
    
    # Melt differences for plotting
    melted_diffs = pd.melt(
        avg_diffs,
        id_vars=["Category"],
        value_vars=["selisih_horizontal", "selisih_vertical", "selisih_diagonal"],
        var_name="Arah Korelasi",
        value_name="Rata-rata Selisih Absolut"
    )
    melted_diffs["Arah Korelasi"] = melted_diffs["Arah Korelasi"].map({
        "selisih_horizontal": "Horizontal",
        "selisih_vertical": "Vertical",
        "selisih_diagonal": "Diagonal"
    })
    
    sns.barplot(
        ax=ax_bar,
        data=melted_diffs,
        x="Category",
        y="Rata-rata Selisih Absolut",
        hue="Arah Korelasi",
        palette="muted"
    )
    ax_bar.set_title("Rata-rata Selisih Absolut Korelasi per Kategori\n(|Optimized - Baseline|)")
    ax_bar.set_xlabel("Kategori Kualitas Citra")
    ax_bar.set_ylabel("Rata-rata Selisih")
    
    # Annotate bar plot with values
    for p in ax_bar.patches:
        height = p.get_height()
        if height > 0:
            ax_bar.annotate(
                f"{height:.5f}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )
            
    plt.tight_layout()
    out_plot_path = OUT_DIR / "summary_correlation_comparison_plot.png"
    plt.savefig(out_plot_path, dpi=300)
    plt.close()
    print(f"[SUKSES] Grafik perbandingan korelasi disimpan di: {out_plot_path}")

if __name__ == "__main__":
    main()
