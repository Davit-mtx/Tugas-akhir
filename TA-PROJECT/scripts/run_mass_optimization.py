import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    output_base_dir = PROJECT_ROOT / "data/results/optimized" 
    
    # Rute Folder Fisik Gambar Optimized
    out_enc_opt_dir = output_base_dir / "enkripsi"
    out_dec_opt_dir = output_base_dir / "dekripsi"
    
    categories = ['high', 'medium', 'low']
    K_HEX = "00112233445566778899aabbccddeeff"

    # Batas Ruang Pencarian HO [r_min, eps, T0, Q]
    lb = [3.5, 0.0001, 100, 10]
    ub = [3.99, 0.05, 2000, 1000]

    # Konfigurasi HO
    SearchAgents = 10
    Max_iterations = 20

    print(f"Konfigurasi HO: {SearchAgents} Agen, {Max_iterations} Iterasi per gambar.\n")

    all_results = []
    all_ho_curves = [] # Menampung kurva konvergensi dari semua gambar

    # 2. Looping per Kategori
    for category in categories:
        input_cat_dir = input_base_dir / category
        
        # Buat Sub-folder Fisik per Kategori untuk gambar
        (out_enc_opt_dir / category).mkdir(parents=True, exist_ok=True)
        (out_dec_opt_dir / category).mkdir(parents=True, exist_ok=True)
        
        if not input_cat_dir.exists():
            continue

        image_files = list(input_cat_dir.glob("*.png")) + list(input_cat_dir.glob("*.jpg"))
        print(f"--- Memproses Kategori: {category.upper()} ({len(image_files)} gambar) ---")

        # 3. Looping per Gambar
        for idx, img_path in enumerate(image_files, 1):
            
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            print(f"  [{idx}/{len(image_files)}] Optimasi & Menyimpan: {img_path.name} ... ", end="", flush=True)

            try:
                # --- A. Jalankan HO untuk mencari parameter X terbaik ---
                def objective_function(X):
                    return evaluate_fitness(img_rgb, K_HEX, X)

                start_opt_time = time.time()
                # Tangkap ho_curve untuk dirata-ratakan nanti
                best_score, best_pos, ho_curve = run_ho(SearchAgents, Max_iterations, lb, ub, objective_function)
                opt_time = time.time() - start_opt_time
                
                all_ho_curves.append(ho_curve)

                # Susun objek konfigurasi dari hasil HO
                opt_cfg = BaselineConfig(
                    r_min=float(best_pos[0]), 
                    eps=float(best_pos[1]), 
                    T0=int(round(best_pos[2])), 
                    Q=int(round(best_pos[3]))
                )

                # --- B. Evaluasi Final Menggunakan Parameter Terbaik ---
                start_time = time.time()
                cipher_opt, _ = encrypt_baseline(img_rgb, K_HEX, cfg=opt_cfg, return_debug=False)
                t_enc = time.time() - start_time

                start_time = time.time()
                decrypted = decrypt_baseline(cipher_opt, K_HEX, cfg=opt_cfg)
                t_dec = time.time() - start_time
                
                # --- Simpan Gambar Fisik (Konversi RGB ke BGR) ---
                cv2.imwrite(str(out_enc_opt_dir / category / img_path.name), cv2.cvtColor(cipher_opt, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_dec_opt_dir / category / img_path.name), cv2.cvtColor(decrypted, cv2.COLOR_RGB2BGR))

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
                print(f"Selesai (Fit: {best_score:.4f}, Entropi: {entropy:.4f})")

            except Exception as e:
                print(f"GAGAL ({e})")

    # 4. Membuat File Rekapitulasi Master & Grafik
    if all_results:
        # Ekspor CSV
        df_master = pd.DataFrame(all_results)
        master_csv_path = output_base_dir / "summary_optimized_results.csv"
        df_master.to_csv(master_csv_path, index=False)
        
        # Cetak Grafik Rata-Rata Konvergensi
        if all_ho_curves:
            # Rata-rata dari seluruh array curve secara vertikal (axis=0)
            avg_curve = np.mean(all_ho_curves, axis=0)
            
            plt.figure(figsize=(8, 6))
            plt.plot(avg_curve, color='#b28d90', linewidth=2, label='Rata-Rata HO')
            plt.title("Rata-Rata Kurva Konvergensi HO (Seluruh Citra)")
            plt.xlabel("Iterasi")
            plt.ylabel("Rata-Rata Skor Fitness Terbaik")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plot_path = output_base_dir / "average_convergence_curve.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
        print(f"\n=== SELESAI! ===")
        print(f"Gambar terenkripsi & terdekripsi tersimpan di: {out_enc_opt_dir} & {out_dec_opt_dir}")
        print(f"Grafik Konvergensi Rata-Rata tersimpan di: {plot_path}")
        print(f"Tabel Master Evaluasi Optimasi tersimpan di: {master_csv_path}")

if __name__ == "__main__":
    main()