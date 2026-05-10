import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set style grafik agar terlihat akademis dan profesional
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def main():
    print("=== MEMBUAT GRAFIK KOMPARASI BASELINE VS HO ===\n")

    baseline_csv = PROJECT_ROOT / "data/results/baseline/summary_all_results.csv"
    optimized_csv = PROJECT_ROOT / "data/results/optimized/summary_optimized_results.csv"
    out_dir = PROJECT_ROOT / "data/results/plots/comparison_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_csv.exists() or not optimized_csv.exists():
        print("[ERROR] File CSV Master tidak ditemukan.")
        return

    # Baca dan gabungkan data
    df_base = pd.read_csv(baseline_csv)
    df_opt = pd.read_csv(optimized_csv)
    df = pd.merge(df_base, df_opt, on=["File Name", "Category"], suffixes=("_base", "_opt"))

    # Hitung rata-rata korelasi absolut per baris
    df['Avg_Corr_base'] = (df['Corr_Horizontal_base'].abs() + df['Corr_Vertical_base'].abs() + df['Corr_Diagonal_base'].abs()) / 3
    df['Avg_Corr_opt'] = (df['Corr_Horizontal_opt'].abs() + df['Corr_Vertical_opt'].abs() + df['Corr_Diagonal_opt'].abs()) / 3

    categories = ['high', 'medium', 'low']
    labels = ['High (DMOS Rendah)', 'Medium', 'Low (DMOS Tinggi)']
    
    # Ambil nilai rata-rata per kategori
    ent_base = [df[df['Category'] == c]['Entropy_base'].mean() for c in categories]
    ent_opt = [df[df['Category'] == c]['Entropy_opt'].mean() for c in categories]
    
    corr_base = [df[df['Category'] == c]['Avg_Corr_base'].mean() for c in categories]
    corr_opt = [df[df['Category'] == c]['Avg_Corr_opt'].mean() for c in categories]

    x = np.arange(len(labels))
    width = 0.35

    # ==========================================
    # 1. GRAFIK ENTROPI (Makin mendekati 8 makin baik)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, ent_base, width, label='Baseline', color='#5DADE2')
    rects2 = ax.bar(x + width/2, ent_opt, width, label='HO Optimized', color='#28B463')

    ax.set_ylabel('Skor Entropi (Bits)')
    ax.set_title('Komparasi Rata-rata Entropi: Baseline vs Optimasi HO')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')

    # Zoom sumbu Y agar perbedaan tervisualisasi (karena nilainya sama-sama di 7.99x)
    min_val = min(min(ent_base), min(ent_opt))
    ax.set_ylim([min_val - 0.0005, 8.0000])

    # Tambahkan label angka di atas batang
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.5f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "1_Entropy_Comparison.png", dpi=300)
    plt.close()
    print("-> Grafik Entropi berhasil disimpan.")

    # ==========================================
    # 2. GRAFIK KORELASI (Makin mendekati 0 makin baik)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, corr_base, width, label='Baseline', color='#E74C3C')
    rects2 = ax.bar(x + width/2, corr_opt, width, label='HO Optimized', color='#F39C12')

    ax.set_ylabel('Korelasi Piksel Absolut')
    ax.set_title('Komparasi Rata-rata Korelasi: Baseline vs Optimasi HO')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Tambahkan label angka
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.5f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "2_Correlation_Comparison.png", dpi=300)
    plt.close()
    print("-> Grafik Korelasi berhasil disimpan.")

    # ==========================================
    # 3. GRAFIK WAKTU KOMPUTASI (BOXPLOT)
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    # Menyiapkan data untuk boxplot
    data_enc_base = df['Enc_Time (s)_base']
    data_enc_opt = df['Enc_Time (s)_opt']
    
    plt.boxplot([data_enc_base, data_enc_opt], labels=['Enkripsi Baseline', 'Enkripsi Param HO'], patch_artist=True, boxprops=dict(facecolor='#D2B4DE'))
    plt.ylabel('Waktu Komputasi (Detik)')
    plt.title('Distribusi Waktu Enkripsi (Tanpa waktu pencarian HO)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_dir / "3_Time_Efficiency.png", dpi=300)
    plt.close()
    print("-> Grafik Waktu Komputasi berhasil disimpan.")

    print(f"\n[SELESAI] Semua grafik telah disimpan di folder: {out_dir}")

if __name__ == "__main__":
    main()
