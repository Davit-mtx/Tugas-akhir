
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Setup Path agar folder src/ terbaca
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.crypto.pipeline import encrypt_baseline, decrypt_baseline, BaselineConfig
from src.metrics.metric import calculate_entropy, calculate_correlation, calculate_npcr_uaci, verify_lossless

def main():
    print("=== PENGUJIAN METRIK EVALUASI KEAMANAN ===")
    
    # 1. Baca Gambar
    img_path = PROJECT_ROOT / "data/samples_30_per_class/high/I04_04_01.png"
    img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_rgb is not None:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    else:
        print("[WARNING] Menggunakan dummy image 128x128.")
        img_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    K = "00112233445566778899aabbccddeeff"
    cfg = BaselineConfig()

    print("\n[PROSES] Melakukan Enkripsi & Dekripsi...")
    
    # Mengukur Waktu Enkripsi
    start_time = time.time()
    cipher_1, _ = encrypt_baseline(img_rgb, K, cfg=cfg, return_debug=False)
    t_enc = time.time() - start_time

    # Mengukur Waktu Dekripsi
    start_time = time.time()
    decrypted = decrypt_baseline(cipher_1, K, cfg=cfg)
    t_dec = time.time() - start_time

    print(f"Waktu Enkripsi : {t_enc:.4f} detik")
    print(f"Waktu Dekripsi : {t_dec:.4f} detik")

    # 2. Persiapan Pengujian NPCR/UACI (Ubah 1 piksel pada plaintext)
    img_rgb_modified = img_rgb.copy()
    img_rgb_modified[0, 0, 0] = (int(img_rgb_modified[0, 0, 0]) + 1) % 256  # Ubah 1 nilai pada Red Channel
    cipher_2, _ = encrypt_baseline(img_rgb_modified, K, cfg=cfg, return_debug=False)

    print("\n=== HASIL EVALUASI METRIK STATISTIK (CHANNEL MERAH/R) ===")
    
    # Ambil channel R (indeks 0) dari Cipher 1 untuk diuji
    R_channel = cipher_1[:, :, 0]
    R_channel_modified = cipher_2[:, :, 0]

    # Evaluasi Entropi
    entropy = calculate_entropy(R_channel)
    print(f"Entropi Informasi   : {entropy:.4f} (Target ~ 8.0)")

    # Evaluasi Korelasi
    corr = calculate_correlation(R_channel)
    print(f"Korelasi Horizontal : {corr['horizontal']:.4f} (Target ~ 0.0)")
    print(f"Korelasi Vertikal   : {corr['vertical']:.4f} (Target ~ 0.0)")
    print(f"Korelasi Diagonal   : {corr['diagonal']:.4f} (Target ~ 0.0)")

    # Evaluasi NPCR & UACI
    npcr, uaci = calculate_npcr_uaci(R_channel, R_channel_modified)
    print(f"NPCR                : {npcr:.4f}% (Target > 99.6%)")
    print(f"UACI                : {uaci:.4f}% (Target ~ 33.4%)")

    # Validasi Dekripsi (Lossless)
    is_lossless = verify_lossless(img_rgb, decrypted)
    print(f"\nStatus Dekripsi     : {'BERHASIL (Lossless)' if is_lossless else 'GAGAL'}")

if __name__ == "__main__":
    main()