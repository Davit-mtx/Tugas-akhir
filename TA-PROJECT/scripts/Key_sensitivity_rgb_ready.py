import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import mesin kriptografi untuk pengujian properti kunci langsung di tempat
from src.crypto.pipeline import encrypt_baseline, decrypt_baseline, BaselineConfig
from src.metrics.metric import calculate_npcr_uaci

CHANNEL_MAP = {
    "R": 0,
    "G": 1,
    "B": 2,
}


def require_columns(df, required_columns, source_name):
    """Validasi sederhana agar script berhenti dengan pesan jelas jika format CSV belum sesuai."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom berikut belum ditemukan pada {source_name}: {missing}\n"
            "Pastikan CSV yang dipakai adalah hasil running terbaru berbasis RGB."
        )


def calculate_npcr_uaci_rgb(cipher_a, cipher_b, prefix="Key"):
    """Menghitung NPCR dan UACI untuk R, G, B, lalu membuat nilai rata-rata RGB."""
    npcr_values = []
    uaci_values = []
    metrics = {}

    for channel_name, channel_idx in CHANNEL_MAP.items():
        npcr_ch, uaci_ch = calculate_npcr_uaci(
            cipher_a[:, :, channel_idx],
            cipher_b[:, :, channel_idx]
        )
        npcr_values.append(npcr_ch)
        uaci_values.append(uaci_ch)

        metrics[f"{prefix}_NPCR_{channel_name} (%)"] = round(npcr_ch, 5)
        metrics[f"{prefix}_UACI_{channel_name} (%)"] = round(uaci_ch, 5)

    metrics[f"{prefix}_NPCR_Avg (%)"] = round(float(np.mean(npcr_values)), 5)
    metrics[f"{prefix}_UACI_Avg (%)"] = round(float(np.mean(uaci_values)), 5)
    return metrics


def get_optional_value(row, column_name):
    """Mengambil nilai kolom jika tersedia; jika tidak tersedia, isi NaN agar CSV tetap rapi."""
    return row[column_name] if column_name in row.index else np.nan


def main():
    print("=== ANALISIS KOMPREHENSIF TUJUAN 3: BASELINE vs OPTIMASI HO BERBASIS RGB ===\n")

    baseline_csv = PROJECT_ROOT / "data/results/baseline/summary_all_results.csv"
    optimized_csv = PROJECT_ROOT / "data/results/optimized/summary_optimized_results.csv"

    if not baseline_csv.exists() or not optimized_csv.exists():
        print("[ERROR] File CSV tidak ditemukan.")
        return

    df_base = pd.read_csv(baseline_csv)
    df_opt = pd.read_csv(optimized_csv)

    # Kolom minimum yang wajib ada pada format summary RGB terbaru.
    require_columns(
        df_base,
        ["File Name", "Category", "Entropy", "Corr_Horizontal", "Corr_Vertical", "Corr_Diagonal"],
        "summary_all_results.csv"
    )
    require_columns(
        df_opt,
        ["File Name", "Category", "Entropy", "Corr_Horizontal", "Corr_Vertical", "Corr_Diagonal",
         "Opt_r_min", "Opt_eps", "Opt_T0", "Opt_Q"],
        "summary_optimized_results.csv"
    )

    df_merged = pd.merge(df_base, df_opt, on=["File Name", "Category"], suffixes=("_base", "_opt"))
    categories = ["high", "medium", "low"]

    # --- BAGIAN 1: ANALISIS STATISTIK, DIFERENSIAL, & WAKTU KOMPUTASI ---
    print("--- 1. KOMPARASI PERFORMA RATA-RATA PER KATEGORI (NILAI AVG RGB) ---")
    for cat in categories:
        df_cat = df_merged[df_merged["Category"] == cat]
        if df_cat.empty:
            continue

        avg_ent_base = df_cat["Entropy_base"].mean()
        avg_ent_opt = df_cat["Entropy_opt"].mean()

        avg_corr_base = (
            df_cat["Corr_Horizontal_base"].abs()
            + df_cat["Corr_Vertical_base"].abs()
            + df_cat["Corr_Diagonal_base"].abs()
        ).mean() / 3
        avg_corr_opt = (
            df_cat["Corr_Horizontal_opt"].abs()
            + df_cat["Corr_Vertical_opt"].abs()
            + df_cat["Corr_Diagonal_opt"].abs()
        ).mean() / 3

        print(f"\nKategori: {cat.upper()} ({len(df_cat)} citra)")
        print(f"  Entropi Baseline AVG RGB   : {avg_ent_base:.5f} -> HO : {avg_ent_opt:.5f}")
        print(f"  Korelasi Baseline AVG RGB  : {avg_corr_base:.5f} -> HO : {avg_corr_opt:.5f}")

        if "NPCR (%)_base" in df_cat.columns and "NPCR (%)_opt" in df_cat.columns:
            print(f"  NPCR Baseline AVG RGB      : {df_cat['NPCR (%)_base'].mean():.5f}% -> HO : {df_cat['NPCR (%)_opt'].mean():.5f}%")
        if "UACI (%)_base" in df_cat.columns and "UACI (%)_opt" in df_cat.columns:
            print(f"  UACI Baseline AVG RGB      : {df_cat['UACI (%)_base'].mean():.5f}% -> HO : {df_cat['UACI (%)_opt'].mean():.5f}%")

    # --- BAGIAN 2: ANALISIS RUANG KUNCI (KEY SPACE) ---
    print("\n--- 2. ANALISIS RUANG KUNCI (KEY SPACE) ---")
    K1_HEX = "00112233445566778899aabbccddeeff"
    panjang_bit = len(K1_HEX) * 4
    print(f"  Kunci Uji        : {K1_HEX}")
    print(f"  Panjang Kunci    : {panjang_bit} bit")
    print(f"  Total Ruang Kunci: 2^{panjang_bit} (Memenuhi standar anti Brute-Force > 2^100)")

    # --- BAGIAN 3: PENCARIAN CITRA JUARA & UJI SENSITIVITAS KUNCI KOMPUTASIONAL ---
    print("\n--- 3. PENCARIAN CITRA TERBAIK & UJI SENSITIVITAS KUNCI RGB ---")

    K2_HEX = "00112233445566778899aabbccddeefe"  # Berbeda 1-bit di ujung (f -> e)
    print(f"  Kunci Asli (K1)  : {K1_HEX}")
    print(f"  Kunci Salah (K2) : {K2_HEX} (Berbeda 1 bit)")

    out_sens_dir = PROJECT_ROOT / "data/results/appendix"
    out_sens_dir.mkdir(parents=True, exist_ok=True)

    top_performers = []

    for cat in categories:
        df_cat = df_merged[df_merged["Category"] == cat]
        if df_cat.empty:
            continue

        # Cari "Citra Juara" berdasarkan nilai AVG RGB Entropy_opt yang paling dekat dengan 8.0.
        best_idx = np.abs(8.0 - df_cat["Entropy_opt"]).idxmin()
        best_row = df_cat.loc[best_idx]

        file_name = best_row["File Name"]
        print(f"\n>> Mengeksekusi Uji Sensitivitas pada Juara {cat.upper()}: {file_name}")

        # Load Gambar Juara
        img_path = PROJECT_ROOT / f"data/samples_30_per_class/{cat}/{file_name}"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"   [SKIP] Gambar tidak dapat dibaca: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Setup Parameter HO hasil optimasi
        opt_cfg = BaselineConfig(
            r_min=float(best_row["Opt_r_min"]),
            eps=float(best_row["Opt_eps"]),
            T0=int(best_row["Opt_T0"]),
            Q=int(best_row["Opt_Q"]),
        )

        # Eksekusi Enkripsi C1 (Kunci Benar) dan C2 (Kunci Salah)
        C1, _ = encrypt_baseline(img_rgb, K1_HEX, cfg=opt_cfg, return_debug=False)
        C2, _ = encrypt_baseline(img_rgb, K2_HEX, cfg=opt_cfg, return_debug=False)

        key_metrics = calculate_npcr_uaci_rgb(C1, C2, prefix="Key")
        print(f"   [Enkripsi] NPCR C1 vs C2 AVG RGB: {key_metrics['Key_NPCR_Avg (%)']:.5f}% (Target > 99%)")
        print(f"   [Enkripsi] UACI C1 vs C2 AVG RGB: {key_metrics['Key_UACI_Avg (%)']:.5f}%")
        print(
            "   [Detail] "
            f"NPCR R/G/B = {key_metrics['Key_NPCR_R (%)']:.5f} / "
            f"{key_metrics['Key_NPCR_G (%)']:.5f} / {key_metrics['Key_NPCR_B (%)']:.5f}"
        )

        # Eksekusi Dekripsi Silang dan Dekripsi Benar
        wrong_dec = decrypt_baseline(C1, K2_HEX, cfg=opt_cfg)
        correct_dec = decrypt_baseline(C1, K1_HEX, cfg=opt_cfg)

        # Simpan gambar bukti fisik ke folder appendix
        base_name = img_path.stem
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_1_Original.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_2_Cipher_K1.png"), cv2.cvtColor(C1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_3_Cipher_K2.png"), cv2.cvtColor(C2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_4_Dec_Wrong.png"), cv2.cvtColor(wrong_dec, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_5_Dec_Correct.png"), cv2.cvtColor(correct_dec, cv2.COLOR_RGB2BGR))
        print("   [Selesai] 5 bukti gambar disimpan ke data/results/appendix/")

        # Simpan metadata dan hasil uji sensitivitas kunci ke CSV rekapan.
        record = {
            "Kategori": cat.upper(),
            "Nama File": file_name,
            "Entropy_Opt_Avg_RGB": round(float(best_row["Entropy_opt"]), 5),
            "Entropy_Opt_R": get_optional_value(best_row, "Entropy_R_opt"),
            "Entropy_Opt_G": get_optional_value(best_row, "Entropy_G_opt"),
            "Entropy_Opt_B": get_optional_value(best_row, "Entropy_B_opt"),
            "NPCR_Opt_Avg_RGB (%)": get_optional_value(best_row, "NPCR (%)_opt"),
            "UACI_Opt_Avg_RGB (%)": get_optional_value(best_row, "UACI (%)_opt"),
            "r_min": best_row["Opt_r_min"],
            "eps": best_row["Opt_eps"],
            "T0": best_row["Opt_T0"],
            "Q": best_row["Opt_Q"],
        }
        record.update(key_metrics)
        top_performers.append(record)

    # Simpan rekap juara berbasis RGB
    pd.DataFrame(top_performers).to_csv(out_sens_dir / "top_performers_rekapan_rgb.csv", index=False)
    print("\n=== SEMUA INDIKATOR TUJUAN 3 TELAH DIEVALUASI BERBASIS RGB ===")


if __name__ == "__main__":
    main()
