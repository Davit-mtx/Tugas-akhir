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

def calculate_rgb_metrics(cipher_original, cipher_modified):
    channel_map = {
        "R": 0,
        "G": 1,
        "B": 2
    }

    entropy_values = []
    corr_h_values = []
    corr_v_values = []
    corr_d_values = []
    npcr_values = []
    uaci_values = []

    rgb_metrics = {}

    for channel_name, channel_idx in channel_map.items():
        channel_original = cipher_original[:, :, channel_idx]
        channel_modified = cipher_modified[:, :, channel_idx]

        entropy_ch = calculate_entropy(channel_original)
        corr_ch = calculate_correlation(channel_original)
        npcr_ch, uaci_ch = calculate_npcr_uaci(channel_original, channel_modified)

        entropy_values.append(entropy_ch)
        corr_h_values.append(corr_ch["horizontal"])
        corr_v_values.append(corr_ch["vertical"])
        corr_d_values.append(corr_ch["diagonal"])
        npcr_values.append(npcr_ch)
        uaci_values.append(uaci_ch)

        rgb_metrics[f"Entropy_{channel_name}"] = round(entropy_ch, 5)
        rgb_metrics[f"Corr_Horizontal_{channel_name}"] = round(corr_ch["horizontal"], 5)
        rgb_metrics[f"Corr_Vertical_{channel_name}"] = round(corr_ch["vertical"], 5)
        rgb_metrics[f"Corr_Diagonal_{channel_name}"] = round(corr_ch["diagonal"], 5)
        rgb_metrics[f"NPCR_{channel_name} (%)"] = round(npcr_ch, 5)
        rgb_metrics[f"UACI_{channel_name} (%)"] = round(uaci_ch, 5)

    rgb_metrics["Entropy"] = round(np.mean(entropy_values), 5)
    rgb_metrics["Corr_Horizontal"] = round(np.mean(corr_h_values), 5)
    rgb_metrics["Corr_Vertical"] = round(np.mean(corr_v_values), 5)
    rgb_metrics["Corr_Diagonal"] = round(np.mean(corr_d_values), 5)
    rgb_metrics["NPCR (%)"] = round(np.mean(npcr_values), 5)
    rgb_metrics["UACI (%)"] = round(np.mean(uaci_values), 5)

    return rgb_metrics

def calculate_rgb_metrics(cipher_original, cipher_modified):
    channel_map = {
        "R": 0,
        "G": 1,
        "B": 2
    }

    entropy_values = []
    corr_h_values = []
    corr_v_values = []
    corr_d_values = []
    npcr_values = []
    uaci_values = []

    rgb_metrics = {}

    for channel_name, channel_idx in channel_map.items():
        channel_original = cipher_original[:, :, channel_idx]
        channel_modified = cipher_modified[:, :, channel_idx]

        entropy_ch = calculate_entropy(channel_original)
        corr_ch = calculate_correlation(channel_original)
        npcr_ch, uaci_ch = calculate_npcr_uaci(channel_original, channel_modified)

        entropy_values.append(entropy_ch)
        corr_h_values.append(corr_ch["horizontal"])
        corr_v_values.append(corr_ch["vertical"])
        corr_d_values.append(corr_ch["diagonal"])
        npcr_values.append(npcr_ch)
        uaci_values.append(uaci_ch)

        rgb_metrics[f"Entropy_{channel_name}"] = round(entropy_ch, 5)
        rgb_metrics[f"Corr_Horizontal_{channel_name}"] = round(corr_ch["horizontal"], 5)
        rgb_metrics[f"Corr_Vertical_{channel_name}"] = round(corr_ch["vertical"], 5)
        rgb_metrics[f"Corr_Diagonal_{channel_name}"] = round(corr_ch["diagonal"], 5)
        rgb_metrics[f"NPCR_{channel_name} (%)"] = round(npcr_ch, 5)
        rgb_metrics[f"UACI_{channel_name} (%)"] = round(uaci_ch, 5)

    rgb_metrics["Entropy"] = round(np.mean(entropy_values), 5)
    rgb_metrics["Corr_Horizontal"] = round(np.mean(corr_h_values), 5)
    rgb_metrics["Corr_Vertical"] = round(np.mean(corr_v_values), 5)
    rgb_metrics["Corr_Diagonal"] = round(np.mean(corr_d_values), 5)
    rgb_metrics["NPCR (%)"] = round(np.mean(npcr_values), 5)
    rgb_metrics["UACI (%)"] = round(np.mean(uaci_values), 5)

    return rgb_metrics

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
    lb = [3.57, 1e-6, 500, 1]
    ub = [3.99, 1e-2, 3000, 256]

    # Konfigurasi HO
    SearchAgents = 10
    Max_iterations = 20

    print(f"Konfigurasi HO: {SearchAgents} Agen, {Max_iterations} Iterasi per gambar.\n")

    master_csv_path = output_base_dir / "summary_optimized_results.csv"
    
    # 2. Fitur Auto-Resume: Cek file yang sudah ada di CSV
    processed_files = set()
    if master_csv_path.exists():
        try:
            df_existing = pd.read_csv(master_csv_path)
            processed_files = set(df_existing["File Name"].tolist())
            print(f"Auto-Resume: Ditemukan {len(processed_files)} file yang sudah diproses di CSV. File-file ini akan dilewati.\n")
        except Exception as e:
            print(f"Gagal membaca CSV untuk resume: {e}")

    all_results = []
    all_ho_curves = [] # Menampung kurva konvergensi dari semua gambar

    # 3. Looping per Kategori
    for category in categories:
        input_cat_dir = input_base_dir / category
        
        # Buat Sub-folder Fisik per Kategori untuk gambar
        (out_enc_opt_dir / category).mkdir(parents=True, exist_ok=True)
        (out_dec_opt_dir / category).mkdir(parents=True, exist_ok=True)
        
        if not input_cat_dir.exists():
            continue

        image_files = list(input_cat_dir.glob("*.png")) + list(input_cat_dir.glob("*.jpg"))
        print(f"--- Memproses Kategori: {category.upper()} ({len(image_files)} gambar) ---")

        # 4. Looping per Gambar
        for idx, img_path in enumerate(image_files, 1):
            
            if img_path.name in processed_files:
                print(f"  [{idx}/{len(image_files)}] Melewati (Skip): {img_path.name} (sudah selesai sebelumnya)")
                continue

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

                # Metrik keamanan
                rgb_metrics = calculate_rgb_metrics(cipher_opt, cipher_mod)

                # --- C. Menyimpan Data ---
                data_row = {
                    "File Name": img_path.name,
                    "Category": category,
                    "Best_Fitness": round(best_score, 6),
                    "Opt_r_min": round(opt_cfg.r_min, 5),
                    "Opt_eps": round(opt_cfg.eps, 5),
                    "Opt_T0": opt_cfg.T0,
                    "Opt_Q": opt_cfg.Q,
                    **rgb_metrics,
                    "Enc_Time (s)": round(t_enc, 4),
                    "Dec_Time (s)": round(t_dec, 4),
                    "HO_Time (s)": round(opt_time, 2),
                    "Lossless": is_lossless
                }

                # Simpan incremental ke CSV agar tidak hilang jika terhenti
                df_row = pd.DataFrame([data_row])
                if not master_csv_path.exists():
                    df_row.to_csv(master_csv_path, index=False)
                else:
                    df_row.to_csv(master_csv_path, mode='a', header=False, index=False)
                
                # Masukkan ke set processed_files agar jika di-run ulang dalam sesi yang sama tidak double
                processed_files.add(img_path.name)

                all_results.append(data_row)
                print(f"Selesai (Fit: {best_score:.4f}, Entropi RGB Avg: {rgb_metrics['Entropy']:.4f})")
            
            except Exception as e:
                print(f"GAGAL ({e})")

    # 5. Membuat Grafik (CSV sudah dibuat otomatis secara incremental)
    if master_csv_path.exists():
        # Cetak Grafik Rata-Rata Konvergensi (dari iterasi terbaru yang dijalankan)
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