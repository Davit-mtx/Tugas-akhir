from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set style grafik agar terlihat akademis dan profesional
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

CATEGORIES = ["high", "medium", "low"]
LABELS = ["High (DMOS Tinggi)", "Medium", "Low (DMOS Rendah)"]
CHANNELS = ["R", "G", "B"]


def require_columns(df, required_columns, source_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom berikut belum ditemukan pada {source_name}: {missing}\n"
            "Pastikan CSV sudah dihasilkan ulang memakai format evaluasi RGB terbaru."
        )


def add_average_abs_correlation(df):
    df["Avg_Corr_base"] = (
        df["Corr_Horizontal_base"].abs()
        + df["Corr_Vertical_base"].abs()
        + df["Corr_Diagonal_base"].abs()
    ) / 3
    df["Avg_Corr_opt"] = (
        df["Corr_Horizontal_opt"].abs()
        + df["Corr_Vertical_opt"].abs()
        + df["Corr_Diagonal_opt"].abs()
    ) / 3
    return df


def mean_by_category(df, column_name):
    return [df[df["Category"] == c][column_name].mean() for c in CATEGORIES]


def annotate_bars(ax, bars, fmt="{:.5f}"):
    for rect in bars:
        height = rect.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def create_category_summary(df, out_dir):
    rows = []
    for category, label in zip(CATEGORIES, LABELS):
        sub = df[df["Category"] == category]
        if sub.empty:
            continue

        row = {
            "Category": category,
            "Label": label,
            "N": len(sub),
            "Entropy_Base_Avg_RGB": round(sub["Entropy_base"].mean(), 5),
            "Entropy_Opt_Avg_RGB": round(sub["Entropy_opt"].mean(), 5),
            "Avg_Corr_Base_RGB": round(sub["Avg_Corr_base"].mean(), 5),
            "Avg_Corr_Opt_RGB": round(sub["Avg_Corr_opt"].mean(), 5),
            "Enc_Time_Base_Mean (s)": round(sub["Enc_Time (s)_base"].mean(), 5),
            "Enc_Time_Opt_Mean (s)": round(sub["Enc_Time (s)_opt"].mean(), 5),
        }

        for metric in ["NPCR (%)", "UACI (%)"]:
            for suffix, label_suffix in [("_base", "Base"), ("_opt", "Opt")]:
                col = metric + suffix
                if col in sub.columns:
                    row[f"{metric.replace(' (%)', '')}_{label_suffix}_Avg_RGB (%)"] = round(sub[col].mean(), 5)

        for channel in CHANNELS:
            for metric in ["Entropy", "NPCR", "UACI"]:
                if metric == "Entropy":
                    base_col = f"Entropy_{channel}_base"
                    opt_col = f"Entropy_{channel}_opt"
                    unit = ""
                else:
                    base_col = f"{metric}_{channel} (%)_base"
                    opt_col = f"{metric}_{channel} (%)_opt"
                    unit = " (%)"

                if base_col in sub.columns and opt_col in sub.columns:
                    row[f"{metric}_Base_{channel}{unit}"] = round(sub[base_col].mean(), 5)
                    row[f"{metric}_Opt_{channel}{unit}"] = round(sub[opt_col].mean(), 5)

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "comparison_category_summary_rgb.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"-> Tabel ringkasan kategori RGB berhasil disimpan: {summary_path.name}")


def plot_entropy(df, out_dir):
    ent_base = mean_by_category(df, "Entropy_base")
    ent_opt = mean_by_category(df, "Entropy_opt")

    x = np.arange(len(LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, ent_base, width, label="Baseline", color="#5DADE2")
    rects2 = ax.bar(x + width / 2, ent_opt, width, label="HO Optimized", color="#28B463")

    ax.set_ylabel("Skor Entropi AVG RGB (Bits)")
    ax.set_title("Komparasi Rata-rata Entropi RGB: Baseline vs Optimasi HO")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend(loc="lower right")

    min_val = min(min(ent_base), min(ent_opt))
    ax.set_ylim([min_val - 0.0005, 8.0000])

    annotate_bars(ax, list(rects1) + list(rects2))
    plt.tight_layout()
    plt.savefig(out_dir / "1_Entropy_Comparison_RGB_Avg.png", dpi=300)
    plt.close()
    print("-> Grafik Entropi AVG RGB berhasil disimpan.")


def plot_correlation(df, out_dir):
    corr_base = mean_by_category(df, "Avg_Corr_base")
    corr_opt = mean_by_category(df, "Avg_Corr_opt")

    x = np.arange(len(LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, corr_base, width, label="Baseline", color="#E74C3C")
    rects2 = ax.bar(x + width / 2, corr_opt, width, label="HO Optimized", color="#F39C12")

    ax.set_ylabel("Korelasi Piksel Absolut AVG RGB")
    ax.set_title("Komparasi Rata-rata Korelasi RGB: Baseline vs Optimasi HO")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend()

    annotate_bars(ax, list(rects1) + list(rects2))
    plt.tight_layout()
    plt.savefig(out_dir / "2_Correlation_Comparison_RGB_Avg.png", dpi=300)
    plt.close()
    print("-> Grafik Korelasi AVG RGB berhasil disimpan.")


def plot_time(df, out_dir):
    plt.figure(figsize=(8, 6))
    data_enc_base = df["Enc_Time (s)_base"]
    data_enc_opt = df["Enc_Time (s)_opt"]

    plt.boxplot(
        [data_enc_base, data_enc_opt],
        labels=["Enkripsi Baseline", "Enkripsi Param HO"],
        patch_artist=True,
        boxprops=dict(facecolor="#D2B4DE"),
    )
    plt.ylabel("Waktu Komputasi (Detik)")
    plt.title("Distribusi Waktu Enkripsi (Tanpa Waktu Pencarian HO)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_dir / "3_Time_Efficiency.png", dpi=300)
    plt.close()
    print("-> Grafik Waktu Komputasi berhasil disimpan.")


def plot_metric_percentage(df, out_dir, metric_name, output_name, title):
    base_col = f"{metric_name}_base"
    opt_col = f"{metric_name}_opt"
    if base_col not in df.columns or opt_col not in df.columns:
        print(f"-> [SKIP] Kolom {metric_name} belum tersedia untuk grafik {title}.")
        return

    base_values = mean_by_category(df, base_col)
    opt_values = mean_by_category(df, opt_col)

    x = np.arange(len(LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, base_values, width, label="Baseline", color="#AF7AC5")
    rects2 = ax.bar(x + width / 2, opt_values, width, label="HO Optimized", color="#45B39D")

    ax.set_ylabel(f"{metric_name} AVG RGB")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend()

    annotate_bars(ax, list(rects1) + list(rects2))
    plt.tight_layout()
    plt.savefig(out_dir / output_name, dpi=300)
    plt.close()
    print(f"-> Grafik {metric_name} AVG RGB berhasil disimpan.")


def plot_channel_entropy(df, out_dir):
    rows = []
    for category, label in zip(CATEGORIES, LABELS):
        sub = df[df["Category"] == category]
        for channel in CHANNELS:
            for method, suffix in [("Baseline", "base"), ("HO Optimized", "opt")]:
                col = f"Entropy_{channel}_{suffix}"
                if col in sub.columns:
                    rows.append({
                        "Category": category,
                        "Label": label,
                        "Channel": channel,
                        "Method": method,
                        "Entropy": sub[col].mean(),
                    })

    if not rows:
        print("-> [SKIP] Kolom Entropy_R/G/B belum tersedia untuk grafik detail kanal.")
        return

    plot_df = pd.DataFrame(rows)
    g = sns.catplot(
        data=plot_df,
        x="Label",
        y="Entropy",
        hue="Method",
        col="Channel",
        kind="bar",
        height=4,
        aspect=0.9,
        sharey=False,
        palette="Set2",
    )
    g.set_axis_labels("Kategori Kualitas", "Entropi per Kanal")
    g.set_titles("Kanal {col_name}")
    g.fig.suptitle("Detail Entropi per Kanal RGB: Baseline vs HO", y=1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "6_Channel_Entropy_Comparison_RGB.png", dpi=300)
    plt.close()
    print("-> Grafik detail Entropi per kanal RGB berhasil disimpan.")


def main():
    print("=== MEMBUAT GRAFIK KOMPARASI BASELINE VS HO BERBASIS RGB ===\n")

    baseline_csv = PROJECT_ROOT / "data/results/baseline/summary_all_results.csv"
    optimized_csv = PROJECT_ROOT / "data/results/optimized/summary_optimized_results.csv"
    out_dir = PROJECT_ROOT / "data/results/plots/comparison_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_csv.exists() or not optimized_csv.exists():
        print("[ERROR] File CSV Master tidak ditemukan.")
        return

    df_base = pd.read_csv(baseline_csv)
    df_opt = pd.read_csv(optimized_csv)

    require_columns(
        df_base,
        ["File Name", "Category", "Entropy", "Corr_Horizontal", "Corr_Vertical", "Corr_Diagonal", "Enc_Time (s)"],
        "summary_all_results.csv",
    )
    require_columns(
        df_opt,
        ["File Name", "Category", "Entropy", "Corr_Horizontal", "Corr_Vertical", "Corr_Diagonal", "Enc_Time (s)"],
        "summary_optimized_results.csv",
    )

    df = pd.merge(df_base, df_opt, on=["File Name", "Category"], suffixes=("_base", "_opt"))
    df = add_average_abs_correlation(df)

    create_category_summary(df, out_dir)
    plot_entropy(df, out_dir)
    plot_correlation(df, out_dir)
    plot_time(df, out_dir)
    plot_metric_percentage(df, out_dir, "NPCR (%)", "4_NPCR_Comparison_RGB_Avg.png", "Komparasi NPCR AVG RGB: Baseline vs Optimasi HO")
    plot_metric_percentage(df, out_dir, "UACI (%)", "5_UACI_Comparison_RGB_Avg.png", "Komparasi UACI AVG RGB: Baseline vs Optimasi HO")
    plot_channel_entropy(df, out_dir)

    print(f"\n[SELESAI] Semua grafik dan tabel komparasi RGB disimpan di folder: {out_dir}")


if __name__ == "__main__":
    main()
