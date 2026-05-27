import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Setup Path agar folder src/ terbaca
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

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

def main():
    print("=== MEMULAI EVALUASI MASSAL (MASS EVALUATION) DENGAN PENYIMPANAN FISIK ===")
    
    # 1. Setup Direktori Input dan Output
    input_base_dir = PROJECT_ROOT / "data/samples_30_per_class"
    output_base_dir = PROJECT_ROOT / "data/results"
    
    # === TAMBAHAN 1: Definisi Rute Folder Fisik ===
    out_enc_base_dir = PROJECT_ROOT / "data/results/baseline/enkripsi"
    out_dec_base_dir = PROJECT_ROOT / "data/results/baseline/dekripsi"
    
    categories = ['high', 'medium', 'low']
    K = "00112233445566778899aabbccddeeff"
    cfg = BaselineConfig()

    # List untuk menyimpan semua hasil demi membuat Master Tabel
    all_results = []

    # 2. Looping per Kategori
    for category in categories:
        input_cat_dir = input_base_dir / category
        output_cat_dir = output_base_dir / category
        
        # # Buat folder output CSV jika belum ada
        # output_cat_dir.mkdir(parents=True, exist_ok=True)
        
        # === TAMBAHAN 2: Buat Sub-folder Fisik per Kategori ===
        (out_enc_base_dir / category).mkdir(parents=True, exist_ok=True)
        (out_dec_base_dir / category).mkdir(parents=True, exist_ok=True)
        
        if not input_cat_dir.exists():
            print(f"[SKIP] Folder tidak ditemukan: {input_cat_dir}")
            continue

        # Ambil semua file gambar (.png, .jpg) di dalam folder kategori
        image_files = list(input_cat_dir.glob("*.png")) + list(input_cat_dir.glob("*.jpg"))
        
        print(f"\n--- Memproses Kategori: {category.upper()} ({len(image_files)} gambar) ---")

        # 3. Looping per Gambar
        for img_path in image_files:
            file_name_awal = img_path.stem  # Mengambil nama file tanpa ekstensi (cth: I04_04_01)
            
            # Baca Gambar
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"  [ERROR] Gagal membaca {img_path.name}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            print(f"  -> Mengevaluasi & Menyimpan: {img_path.name} ... ", end="", flush=True)

            try:
                # --- A. Mengukur Waktu Enkripsi & Dekripsi ---
                start_time = time.time()
                cipher_1, _ = encrypt_baseline(img_rgb, K, cfg=cfg, return_debug=False)
                t_enc = time.time() - start_time

                start_time = time.time()
                decrypted = decrypt_baseline(cipher_1, K, cfg=cfg)
                t_dec = time.time() - start_time

                # === TAMBAHAN 3: Simpan Gambar Fisik (Konversi RGB ke BGR) ===
                cv2.imwrite(str(out_enc_base_dir / category / img_path.name), cv2.cvtColor(cipher_1, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_dec_base_dir / category / img_path.name), cv2.cvtColor(decrypted, cv2.COLOR_RGB2BGR))

                # --- B. Memverifikasi Lossless ---
                is_lossless = verify_lossless(img_rgb, decrypted)

                # --- C. Persiapan NPCR/UACI (Ubah 1 piksel pada Red Channel plaintext) ---
                img_rgb_modified = img_rgb.copy()
                pixel_awal = int(img_rgb_modified[0, 0, 0])
                img_rgb_modified[0, 0, 0] = pixel_awal ^ 1
                
                cipher_2, _ = encrypt_baseline(img_rgb_modified, K, cfg=cfg, return_debug=False)

                # --- D. Menghitung Metrik Keamanan (Fokus Channel R untuk evaluasi) ---
                rgb_metrics = calculate_rgb_metrics(cipher_1, cipher_2)

                # --- E. Menyimpan Data ke Dictionary ---
                data_row = {
                    "File Name": img_path.name,
                    "Category": category,
                    **rgb_metrics,
                    "Enc_Time (s)": round(t_enc, 4),
                    "Dec_Time (s)": round(t_dec, 4),
                    "Lossless": is_lossless
                }

                # Simpan ke Master List
                all_results.append(data_row)

                # # 4. Export CSV Spesifik per Gambar
                # df_single = pd.DataFrame([data_row])
                # out_csv_path = output_cat_dir / f"result.{file_name_awal}.csv"
                # df_single.to_csv(out_csv_path, index=False)

                print("Selesai")

            except Exception as e:
                print(f"GAGAL ({e})")

    # 5. Membuat File Rekapitulasi Master
    if all_results:
        df_master = pd.DataFrame(all_results)
        master_csv_path = output_base_dir / "baseline/summary_all_results.csv"
        df_master.to_csv(master_csv_path, index=False)
        print(f"\n=== SELESAI! ===")
        print(f"Gambar fisik tersimpan di: {out_enc_base_dir} dan {out_dec_base_dir}")
        print(f"Tabel Rekapitulasi Master tersimpan di: {master_csv_path}")
    else:
        print("\n[WARNING] Tidak ada data yang berhasil dievaluasi.")

if __name__ == "__main__":
    main()