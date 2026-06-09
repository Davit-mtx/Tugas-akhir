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
    print("=== PROSES SUMMARY NILAI NPCR & UACI (BASELINE VS OPTIMIZED) ===")
    
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
    req_cols = [
        "File Name", "Category", 
        "NPCR_R (%)", "UACI_R (%)", 
        "NPCR_G (%)", "UACI_G (%)", 
        "NPCR_B (%)", "UACI_B (%)", 
        "NPCR (%)", "UACI (%)"
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
        "NPCR_R (%)": "NPCR_R_baseline",
        "UACI_R (%)": "UACI_R_baseline",
        "NPCR_G (%)": "NPCR_G_baseline",
        "UACI_G (%)": "UACI_G_baseline",
        "NPCR_B (%)": "NPCR_B_baseline",
        "UACI_B (%)": "UACI_B_baseline",
        "NPCR (%)": "NPCR_baseline",
        "UACI (%)": "UACI_baseline"
    }
    
    opt_rename = {
        "NPCR_R (%)": "NPCR_R_optimized",
        "UACI_R (%)": "UACI_R_optimized",
        "NPCR_G (%)": "NPCR_G_optimized",
        "UACI_G (%)": "UACI_G_optimized",
        "NPCR_B (%)": "NPCR_B_optimized",
        "UACI_B (%)": "UACI_B_optimized",
        "NPCR (%)": "NPCR_optimized",
        "UACI (%)": "UACI_optimized"
    }
    
    df_base_filtered = df_base_filtered.rename(columns=base_rename)
    df_opt_filtered = df_opt_filtered.rename(columns=opt_rename)
    
    # Merge datasets on File Name and Category
    merged_df = pd.merge(df_base_filtered, df_opt_filtered, on=["File Name", "Category"])
    
    # Calculate difference (selisih) of overall average NPCR & UACI (optimized - baseline)
    merged_df["selisih_npcr"] = merged_df["NPCR_optimized"] - merged_df["NPCR_baseline"]
    merged_df["selisih_uaci"] = merged_df["UACI_optimized"] - merged_df["UACI_baseline"]
    
    # Reorder columns for readability
    col_order = [
        "File Name", "Category",
        "NPCR_R_baseline", "NPCR_G_baseline", "NPCR_B_baseline", "NPCR_baseline",
        "NPCR_R_optimized", "NPCR_G_optimized", "NPCR_B_optimized", "NPCR_optimized",
        "selisih_npcr",
        "UACI_R_baseline", "UACI_G_baseline", "UACI_B_baseline", "UACI_baseline",
        "UACI_R_optimized", "UACI_G_optimized", "UACI_B_optimized", "UACI_optimized",
        "selisih_uaci"
    ]
    merged_df = merged_df[col_order]
    
    # Save to CSV in results/plots/
    out_csv_path = OUT_DIR / "summary_npcr_uaci.csv"
    merged_df.to_csv(out_csv_path, index=False)
    print(f"[SUKSES] File summary CSV disimpan di: {out_csv_path}")
    
    # Setup plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # ==================== PLOT NPCR ====================
    fig_npcr, axes_npcr = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Boxplot NPCR baseline vs optimized
    melted_npcr = pd.melt(
        merged_df, 
        id_vars=["Category"], 
        value_vars=["NPCR_baseline", "NPCR_optimized"],
        var_name="Metode", 
        value_name="NPCR"
    )
    melted_npcr["Metode"] = melted_npcr["Metode"].map({
        "NPCR_baseline": "Baseline",
        "NPCR_optimized": "Optimized"
    })
    
    sns.boxplot(
        ax=axes_npcr[0],
        data=melted_npcr,
        x="Category",
        y="NPCR",
        hue="Metode",
        palette="Set2"
    )
    axes_npcr[0].set_title("Distribusi Nilai NPCR: Baseline vs Optimized")
    axes_npcr[0].set_xlabel("Kategori Kualitas Image")
    axes_npcr[0].set_ylabel("Nilai NPCR (%)")
    
    # Right subplot: Barplot average difference NPCR
    category_npcr_diffs = merged_df.groupby("Category")["selisih_npcr"].mean().reset_index()
    sns.barplot(
        ax=axes_npcr[1],
        data=category_npcr_diffs,
        x="Category",
        y="selisih_npcr",
        hue="Category",
        legend=False,
        palette="coolwarm"
    )
    axes_npcr[1].set_title("Rata-rata Selisih NPCR per Kategori\n(Optimized - Baseline)")
    axes_npcr[1].set_xlabel("Kategori Kualitas Image")
    axes_npcr[1].set_ylabel("Rata-rata Selisih NPCR (%)")
    
    # Add values on top of the bars
    for p in axes_npcr[1].patches:
        height = p.get_height()
        axes_npcr[1].annotate(
            f"{height:+.6f}%",
            xy=(p.get_x() + p.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        
    plt.tight_layout()
    out_npcr_plot = OUT_DIR / "summary_npcr_comparison_plot.png"
    plt.savefig(out_npcr_plot, dpi=300)
    plt.close()
    print(f"[SUKSES] Grafik NPCR disimpan di: {out_npcr_plot}")
    
    # ==================== PLOT UACI ====================
    fig_uaci, axes_uaci = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Boxplot UACI baseline vs optimized
    melted_uaci = pd.melt(
        merged_df, 
        id_vars=["Category"], 
        value_vars=["UACI_baseline", "UACI_optimized"],
        var_name="Metode", 
        value_name="UACI"
    )
    melted_uaci["Metode"] = melted_uaci["Metode"].map({
        "UACI_baseline": "Baseline",
        "UACI_optimized": "Optimized"
    })
    
    sns.boxplot(
        ax=axes_uaci[0],
        data=melted_uaci,
        x="Category",
        y="UACI",
        hue="Metode",
        palette="Set2"
    )
    axes_uaci[0].set_title("Distribusi Nilai UACI: Baseline vs Optimized")
    axes_uaci[0].set_xlabel("Kategori Kualitas Image")
    axes_uaci[0].set_ylabel("Nilai UACI (%)")
    
    # Right subplot: Barplot average difference UACI
    category_uaci_diffs = merged_df.groupby("Category")["selisih_uaci"].mean().reset_index()
    sns.barplot(
        ax=axes_uaci[1],
        data=category_uaci_diffs,
        x="Category",
        y="selisih_uaci",
        hue="Category",
        legend=False,
        palette="coolwarm"
    )
    axes_uaci[1].set_title("Rata-rata Selisih UACI per Kategori\n(Optimized - Baseline)")
    axes_uaci[1].set_xlabel("Kategori Kualitas Image")
    axes_uaci[1].set_ylabel("Rata-rata Selisih UACI (%)")
    
    # Add values on top of the bars
    for p in axes_uaci[1].patches:
        height = p.get_height()
        axes_uaci[1].annotate(
            f"{height:+.6f}%",
            xy=(p.get_x() + p.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        
    plt.tight_layout()
    out_uaci_plot = OUT_DIR / "summary_uaci_comparison_plot.png"
    plt.savefig(out_uaci_plot, dpi=300)
    plt.close()
    print(f"[SUKSES] Grafik UACI disimpan di: {out_uaci_plot}")

if __name__ == "__main__":
    main()
