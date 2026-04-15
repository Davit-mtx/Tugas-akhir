import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Setup Path agar folder src/ terbaca
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ho.optimizer import run_ho
from src.ho.fitness import evaluate_fitness
from src.crypto.pipeline import encrypt_baseline, decrypt_baseline, BaselineConfig
from src.metrics.metric import calculate_entropy, calculate_correlation, calculate_npcr_uaci, verify_lossless

def main():
    print("=== MEMULAI OPTIMASI MASSAL (MASS OPTIMIZATION) DENGAN HO ===")
    
    # 1. Setup Direktori Input dan Output
    input_base_dir = PROJECT_ROOT / "data/samples_30_per_class"
    output_base_dir = PROJECT_ROOT / "data/results/optimized" # Folder khusus hasil optimasi
    
    categories = ['high', 'medium', 'low']
    K_HEX = "00112233445566778899aabbccddeeff"

    # Batas Ruang Pencarian HO [r_min, eps, T0, Q]
    lb = [3.5, 0.0001, 100, 10]
    ub = [3.99, 0.05, 2000, 1000]

    # Konfigurasi HO (Ubah jika Anda akan melakukan running semalaman)
    SearchAgents = 10
    Max_iterations = 20

    print(f"Konfigurasi HO: {SearchAgents} Agen, {Max_iterations} Iterasi per gambar.\n")

    all_results = []

    # 2. Looping per Kategori
    for category in categories:
        input_cat_dir = input_base_dir / category
        output_cat_dir = output_base_dir / category
        
        output_cat_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_cat_dir.exists():
            continue

        image_files = list(input_cat_dir.glob("*.png")) + list(input_cat_dir.glob("*.jpg"))
        print(f"--- Memproses Kategori: {category.upper()} ({len(image_files)} gambar) ---")

        # 3. Looping per Gambar
        for idx, img_path in enumerate(image_files, 1):
            file_name_awal = img_path.stem
            
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            print(f"  [{idx}/{len(image_files)}] Optimasi: {img_path.name} ... ", end="", flush=True)

            try:
                # --- A. Jalankan HO untuk mencari parameter X terbaik ---
                def objective_function(X):
                    return evaluate_fitness(img_rgb, K_HEX, X)

                start_opt_time = time.time()
                best_score, best_pos, _ = run_ho(SearchAgents, Max_iterations, lb, ub, objective_function)
                opt_time = time.time() - start_opt_time

                # Susun objek konfigurasi dari hasil HO
                opt_cfg = BaselineConfig(
                    r_min=float(best_pos[0]), 
                    eps=float(best_pos[1]), 
                    T0=int(round(best_pos[2])), 
                    Q=int(round(best_pos[3]))
                )

                # --- B. Evaluasi Final Menggunakan Parameter Terbaik ---
                # Mengukur Waktu Enkripsi & Dekripsi
                start_time = time.time()
                cipher_opt, _ = encrypt_baseline(img_rgb, K_HEX, cfg=opt_cfg, return_debug=False)
                t_enc = time.time() - start_time

                start_time = time.time()
                decrypted = decrypt_baseline(cipher_opt, K_HEX, cfg=opt_cfg)
                t_dec = time.time() - start_time

                # Validasi Lossless
                is_lossless = verify_lossless(img_rgb, decrypted)

                # Persiapan NPCR/UACI (XOR 1 bit)
                img_rgb_modified = img_rgb.copy()
                pixel_awal = int(img_rgb_modified[0, 0, 0])
                img_rgb_modified[0, 0, 0] = pixel_awal ^ 1
                cipher_mod, _ = encrypt_baseline(img_rgb_modified, K_HEX, cfg=opt_cfg, return_debug=False)

                # Menghitung Metrik Keamanan (Channel R)
                R_channel = cipher_opt[:, :, 0]
                R_channel_modified = cipher_mod[:, :, 0]

                entropy = calculate_entropy(R_channel)
                corr = calculate_correlation(R_channel)
                npcr, uaci = calculate_npcr_uaci(R_channel, R_channel_modified)

                # --- C. Menyimpan Data ---
                data_row = {
                    "File Name": img_path.name,
                    "Category": category,
                    "Best_Fitness": round(best_score, 6),
                    "Opt_r_min": round(opt_cfg.r_min, 5),
                    "Opt_eps": round(opt_cfg.eps, 5),
                    "Opt_T0": opt_cfg.T0,
                    "Opt_Q": opt_cfg.Q,
                    "Entropy": round(entropy, 5),
                    "Corr_Horizontal": round(corr['horizontal'], 5),
                    "Corr_Vertical": round(corr['vertical'], 5),
                    "Corr_Diagonal": round(corr['diagonal'], 5),
                    "NPCR (%)": round(npcr, 5),
                    "UACI (%)": round(uaci, 5),
                    "Enc_Time (s)": round(t_enc, 4),
                    "Dec_Time (s)": round(t_dec, 4),
                    "HO_Time (s)": round(opt_time, 2),
                    "Lossless": is_lossless
                }

                all_results.append(data_row)

                # Export CSV Spesifik
                df_single = pd.DataFrame([data_row])
                out_csv_path = output_cat_dir / f"result.{file_name_awal}.csv"
                df_single.to_csv(out_csv_path, index=False)

                print(f"Selesai (Fit: {best_score:.4f}, Entropi: {entropy:.4f})")

            except Exception as e:
                print(f"GAGAL ({e})")

    # 4. Membuat File Rekapitulasi Master
    if all_results:
        df_master = pd.DataFrame(all_results)
        master_csv_path = output_base_dir / "summary_optimized_results.csv"
        df_master.to_csv(master_csv_path, index=False)
        print(f"\n=== SELESAI! ===")
        print(f"Tabel Master Evaluasi Optimasi tersimpan di: {master_csv_path}")

if __name__ == "__main__":
    main()