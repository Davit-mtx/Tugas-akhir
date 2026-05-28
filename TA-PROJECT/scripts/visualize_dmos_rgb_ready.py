from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set style grafik untuk standar publikasi/skripsi
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

CATEGORIES = ["high", "medium", "low"]
CHANNELS = ["R", "G", "B"]


def require_columns(df, required_columns, source_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom berikut belum ditemukan pada {source_name}: {missing}\n"
            "Pastikan CSV optimized sudah dihasilkan ulang memakai format evaluasi RGB terbaru."
        )


def add_avg_correlation(df):
    df["Avg_Corr"] = (
        df["Corr_Horizontal"].abs()
        + df["Corr_Vertical"].abs()
        + df["Corr_Diagonal"].abs()
    ) / 3
    return df


def build_stat_summary(df, out_dir):
    candidate_columns = [
        "Opt_r_min", "Opt_eps", "Opt_T0", "Opt_Q",
        "Entropy", "Entropy_R", "Entropy_G", "Entropy_B",
        "Corr_Horizontal", "Corr_Vertical", "Corr_Diagonal", "Avg_Corr",
        "NPCR (%)", "NPCR_R (%)", "NPCR_G (%)", "NPCR_B (%)",
        "UACI (%)", "UACI_R (%)", "UACI_G (%)", "UACI_B (%)",
        "Enc_Time (s)", "Dec_Time (s)", "HO_Time (s)",
    ]
    agg_dict = {col: ["mean", "std"] for col in candidate_columns if col in df.columns}

    stat_summary = df.groupby("Category", observed=False).agg(agg_dict).round(5)
    stat_csv_path = out_dir / "dmos_statistical_summary_rgb.csv"
    stat_summary.to_csv(stat_csv_path)
    print(f"   [OK] Tabel statistik RGB disimpan di: {stat_csv_path.name}")


def plot_chaos_parameters(df, out_dir):
    print("-> Membuat grafik persebaran parameter Chaos...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=df, x="Label", y="Opt_r_min", ax=axes[0], palette="Blues")
    axes[0].set_title("Distribusi Parameter r_min berdasarkan Kualitas Citra")
    axes[0].set_ylabel("Nilai r_min Teroptimasi")
    axes[0].set_xlabel("Kategori Kualitas (DMOS)")

    sns.boxplot(data=df, x="Label", y="Opt_eps", ax=axes[1], palette="Greens")
    axes[1].set_title("Distribusi Parameter Epsilon berdasarkan Kualitas Citra")
    axes[1].set_ylabel("Nilai Epsilon Teroptimasi")
    axes[1].set_xlabel("Kategori Kualitas (DMOS)")

    plt.tight_layout()
    plt.savefig(out_dir / "1_Chaos_Parameters_DMOS.png", dpi=300)
    plt.close()


def plot_iteration_parameters(df, out_dir):
    print("-> Membuat grafik beban komputasi (Iterasi)...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=df, x="Label", y="Opt_T0", ax=axes[0], palette="Oranges")
    axes[0].set_title("Distribusi Iterasi Transient (T0) berdasarkan Kualitas Citra")
    axes[0].set_ylabel("Jumlah Iterasi (T0)")
    axes[0].set_xlabel("Kategori Kualitas (DMOS)")

    sns.boxplot(data=df, x="Label", y="Opt_Q", ax=axes[1], palette="Purples")
    axes[1].set_title("Distribusi Faktor Kuantisasi (Q) berdasarkan Kualitas Citra")
    axes[1].set_ylabel("Nilai Q")
    axes[1].set_xlabel("Kategori Kualitas (DMOS)")

    plt.tight_layout()
    plt.savefig(out_dir / "2_Iteration_Parameters_DMOS.png", dpi=300)
    plt.close()


def plot_entropy_consistency(df, out_dir):
    print("-> Membuat grafik konsistensi keamanan AVG RGB...")
    plt.figure(figsize=(8, 6))

    sns.violinplot(data=df, x="Label", y="Entropy", inner=None, color=".8", alpha=0.5)
    sns.stripplot(data=df, x="Label", y="Entropy", size=6, jitter=True, palette="Dark2", alpha=0.7)

    plt.title("Konsistensi Keamanan Entropi AVG RGB Lintas Kualitas Citra")
    plt.ylabel("Skor Entropi AVG RGB (Mendekati 8 = Baik)")
    plt.xlabel("Kategori Kualitas (DMOS)")
    plt.axhline(y=7.9990, color="r", linestyle="--", alpha=0.5, label="Target > 7.999")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "3_Security_Consistency_DMOS_RGB_Avg.png", dpi=300)
    plt.close()


def plot_channel_entropy(df, out_dir):
    if not all(f"Entropy_{channel}" in df.columns for channel in CHANNELS):
        print("-> [SKIP] Kolom Entropy_R/G/B belum lengkap, grafik entropi per kanal tidak dibuat.")
        return

    print("-> Membuat grafik entropi per kanal RGB...")
    rows = []
    for _, row in df.iterrows():
        for channel in CHANNELS:
            rows.append({
                "Label": row["Label"],
                "Channel": channel,
                "Entropy": row[f"Entropy_{channel}"],
            })

    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=plot_df, x="Label", y="Entropy", hue="Channel", palette="Set2")
    plt.title("Distribusi Entropi per Kanal RGB berdasarkan Kualitas Citra")
    plt.ylabel("Skor Entropi per Kanal")
    plt.xlabel("Kategori Kualitas (DMOS)")
    plt.tight_layout()
    plt.savefig(out_dir / "4_Channel_Entropy_DMOS_RGB.png", dpi=300)
    plt.close()


def plot_npcr_uaci_consistency(df, out_dir):
    available_metrics = [metric for metric in ["NPCR (%)", "UACI (%)", "Avg_Corr"] if metric in df.columns]
    if not available_metrics:
        print("-> [SKIP] Kolom NPCR/UACI/Avg_Corr belum tersedia untuk grafik tambahan.")
        return

    print("-> Membuat grafik konsistensi NPCR, UACI, dan korelasi AVG RGB...")
    for metric in available_metrics:
        safe_name = metric.replace(" (%)", "").replace(" ", "_").replace("/", "_")
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="Label", y=metric, palette="Set3")
        sns.stripplot(data=df, x="Label", y=metric, color="black", size=4, jitter=True, alpha=0.5)
        plt.title(f"Distribusi {metric} AVG RGB berdasarkan Kualitas Citra")
        plt.ylabel(metric)
        plt.xlabel("Kategori Kualitas (DMOS)")
        plt.tight_layout()
        plt.savefig(out_dir / f"5_{safe_name}_DMOS_RGB_Avg.png", dpi=300)
        plt.close()


def main():
    print("=== MENGANALISIS PENGARUH DMOS (TUJUAN 4) BERBASIS RGB ===\n")

    opt_csv_path = PROJECT_ROOT / "data/results/optimized/summary_optimized_results.csv"
    out_dir = PROJECT_ROOT / "data/results/plots/dmos_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not opt_csv_path.exists():
        print("[ERROR] File summary_optimized_results.csv tidak ditemukan!")
        return

    df = pd.read_csv(opt_csv_path)
    require_columns(
        df,
        ["Category", "Opt_r_min", "Opt_eps", "Opt_T0", "Opt_Q", "Entropy", "Corr_Horizontal", "Corr_Vertical", "Corr_Diagonal"],
        "summary_optimized_results.csv",
    )

    df = add_avg_correlation(df)

    # Pastikan urutan kategori benar (High -> Medium -> Low)
    df["Category"] = pd.Categorical(df["Category"], categories=CATEGORIES, ordered=True)

    # Pemetaan label untuk grafik (KADID-10k: High = Bagus/DMOS Rendah, Low = Rusak/DMOS Tinggi)
    label_map = {"high": "High (Bagus/DMOS Rendah)", "medium": "Medium", "low": "Low (Rusak/DMOS Tinggi)"}
    df["Label"] = df["Category"].map(label_map)

    print("-> Menghitung tabel statistik RGB...")
    build_stat_summary(df, out_dir)

    plot_chaos_parameters(df, out_dir)
    plot_iteration_parameters(df, out_dir)
    plot_entropy_consistency(df, out_dir)
    plot_channel_entropy(df, out_dir)
    plot_npcr_uaci_consistency(df, out_dir)

    print("\n=== SELESAI! ===")
    print(f"Semua grafik dan tabel untuk Tujuan 4 berbasis RGB telah disimpan di: {out_dir}")


if __name__ == "__main__":
    main()
