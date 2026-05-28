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

def main():
    print("=== ANALISIS KOMPREHENSIF TUJUAN 3: BASELINE vs OPTIMASI HO ===\n")

    baseline_csv = PROJECT_ROOT / "data/results/baseline/summary_all_results.csv"
    optimized_csv = PROJECT_ROOT / "data/results/optimized/summary_optimized_results.csv"

    if not baseline_csv.exists() or not optimized_csv.exists():
        print("[ERROR] File CSV tidak ditemukan.")
        return

    df_base = pd.read_csv(baseline_csv)
    df_opt = pd.read_csv(optimized_csv)
    df_merged = pd.merge(df_base, df_opt, on=["File Name", "Category"], suffixes=("_base", "_opt"))
    categories = ['high', 'medium', 'low']

    # --- BAGIAN 1: ANALISIS STATISTIK, DIFERENSIAL, & WAKTU KOMPUTASI ---
    print("--- 1. KOMPARASI PERFORMA RATA-RATA PER KATEGORI ---")
    for cat in categories:
        df_cat = df_merged[df_merged['Category'] == cat]
        if df_cat.empty: continue
            
        avg_ent_base = df_cat['Entropy_base'].mean()
        avg_ent_opt = df_cat['Entropy_opt'].mean()
        
        avg_corr_base = (df_cat['Corr_Horizontal_base'].abs() + df_cat['Corr_Vertical_base'].abs() + df_cat['Corr_Diagonal_base'].abs()).mean() / 3
        avg_corr_opt = (df_cat['Corr_Horizontal_opt'].abs() + df_cat['Corr_Vertical_opt'].abs() + df_cat['Corr_Diagonal_opt'].abs()).mean() / 3
        
        print(f"\nKategori: {cat.upper()} ({len(df_cat)} citra)")
        print(f"  Entropi Baseline   : {avg_ent_base:.5f} -> HO : {avg_ent_opt:.5f}")
        print(f"  Korelasi Baseline  : {avg_corr_base:.5f} -> HO : {avg_corr_opt:.5f}")

    # --- BAGIAN 2: ANALISIS RUANG KUNCI (KEY SPACE) ---
    print("\n--- 2. ANALISIS RUANG KUNCI (KEY SPACE) ---")
    K1_HEX = "00112233445566778899aabbccddeeff"
    panjang_bit = len(K1_HEX) * 4
    print(f"  Kunci Uji        : {K1_HEX}")
    print(f"  Panjang Kunci    : {panjang_bit} bit")
    print(f"  Total Ruang Kunci: 2^{panjang_bit} (Memenuhi standar anti Brute-Force > 2^100)")

    # --- BAGIAN 3: PENCARIAN CITRA JUARA & UJI SENSITIVITAS KUNCI KOMPUTASIONAL ---
    print("\n--- 3. PENCARIAN CITRA TERBAIK & UJI SENSITIVITAS KUNCI ---")
    
    K2_HEX = "00112233445566778899aabbccddeefe" # Berbeda 1-bit di ujung (f -> e)
    print(f"  Kunci Asli (K1)  : {K1_HEX}")
    print(f"  Kunci Salah (K2) : {K2_HEX} (Berbeda 1 bit)")

    out_sens_dir = PROJECT_ROOT / "data/results/appendix"
    out_sens_dir.mkdir(parents=True, exist_ok=True)

    top_performers = []

    for cat in categories:
        df_cat = df_merged[df_merged['Category'] == cat]
        if df_cat.empty: continue
            
        # Cari "Citra Juara" berdasarkan jarak terdekat dengan entropi absolut 8.0
        best_idx = np.abs(8.0 - df_cat['Entropy_opt']).idxmin()
        best_row = df_cat.loc[best_idx]
        
        file_name = best_row['File Name']
        print(f"\n>> Mengeksekusi Uji Sensitivitas pada Juara {cat.upper()}: {file_name}")
        
        # Simpan metadata untuk rekapan
        top_performers.append({
            "Kategori": cat.upper(), "Nama File": file_name, "Entropi": best_row['Entropy_opt'],
            "r_min": best_row['Opt_r_min'], "eps": best_row['Opt_eps'], 
            "T0": best_row['Opt_T0'], "Q": best_row['Opt_Q']
        })

        # Load Gambar Juara
        img_path = PROJECT_ROOT / f"data/samples_30_per_class/{cat}/{file_name}"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Setup Parameter HO hasil optimasi
        opt_cfg = BaselineConfig(r_min=best_row['Opt_r_min'], eps=best_row['Opt_eps'], 
                                 T0=int(best_row['Opt_T0']), Q=int(best_row['Opt_Q']))

        # Eksekusi Enkripsi C1 (Kunci Benar) dan C2 (Kunci Salah)
        C1, _ = encrypt_baseline(img_rgb, K1_HEX, cfg=opt_cfg, return_debug=False)
        C2, _ = encrypt_baseline(img_rgb, K2_HEX, cfg=opt_cfg, return_debug=False)
        
        npcr, uaci = calculate_npcr_uaci(C1[:,:,0], C2[:,:,0])
        print(f"   [Enkripsi] NPCR C1 vs C2: {npcr:.5f}% (Target > 99%)")
        print(f"   [Enkripsi] UACI C1 vs C2: {uaci:.5f}%")

        # Eksekusi Dekripsi Silang (C1 didekripsi pakai K2)
        wrong_dec = decrypt_baseline(C1, K2_HEX, cfg=opt_cfg)

        # Dekripsi Benar (C1 didekripsi pakai K1 -> Pasti Berhasil)
        correct_dec = decrypt_baseline(C1, K1_HEX, cfg=opt_cfg)
        
        # Simpan 4 gambar bukti fisik ke folder appendix
        base_name = img_path.stem
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_1_Original.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_2_Cipher_K1.png"), cv2.cvtColor(C1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_3_Cipher_K2.png"), cv2.cvtColor(C2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_4_Dec_Wrong.png"), cv2.cvtColor(wrong_dec, cv2.COLOR_RGB2BGR))

        # --- TAMBAHKAN BARIS INI: Simpan Hasil Dekripsi Benar ---
        cv2.imwrite(str(out_sens_dir / f"{cat}_{base_name}_5_Dec_Correct.png"), cv2.cvtColor(correct_dec, cv2.COLOR_RGB2BGR))
        print(f"   [Selesai] 5 Bukti gambar disimpan ke data/results/appendix/")

    # Simpan rekap juara
    pd.DataFrame(top_performers).to_csv(out_sens_dir / "top_performers_rekapan.csv", index=False)
    print("\n=== SEMUA INDIKATOR TUJUAN 3 TELAH DIEVALUASI SEPENUHNYA ===")

if __name__ == "__main__":
    main()