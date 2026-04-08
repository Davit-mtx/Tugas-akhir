from pathlib import Path
import cv2
import numpy as np

from src.crypto.pipeline import encrypt_baseline, decrypt_baseline, BaselineConfig

PROJECT_ROOT = Path(__file__).resolve().parent

# 1. Setup Data dan Kunci
img_path = PROJECT_ROOT / "data/samples_30_per_class/high/I04_04_01.png"
img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

if img_bgr is None:
    # Dummy data jika file tidak ditemukan untuk keperluan testing cepat
    print("Warning: Gambar asli tidak ditemukan. Menggunakan dummy image 128x128.")
    img_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
else:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

K = "00112233445566778899aabbccddeeff"
cfg = BaselineConfig(r_min=3.70, eps=0.002, T0=500, Q=256)

print("=== PENGUJIAN FULL PIPELINE (ENKRIPSI & DEKRIPSI) ===")

# 2. Proses Enkripsi
print(f"1. Mengenkripsi citra berukuran {img_rgb.shape}...")
cipher, _ = encrypt_baseline(img_rgb, K, cfg=cfg, return_debug=False)

# 3. Proses Dekripsi
print("2. Memulai dekripsi invers deterministik...")
decrypted = decrypt_baseline(cipher, K, cfg=cfg)

# 4. Validasi Exact Match (Syarat Mutlak Proposal)
print("\n=== HASIL EVALUASI ===")
is_lossless = np.array_equal(img_rgb, decrypted)

if is_lossless:
    print("[SUCCESS] P == P^. Dekripsi berhasil 100% tanpa kehilangan data (Lossless)!")
else:
    print("[FAILED] Hasil dekripsi tidak identik dengan citra asli.")
    # Menghitung seberapa banyak pixel yang miss
    diff = np.sum(img_rgb != decrypted)
    print(f"Terdapat perbedaan pada {diff} elemen warna piksel.")

# 5. Export Hasil (Opsional)
out_cipher_path = Path("data") / "test_cipher.png"
out_dec_path = Path("data") / "test_decrypted.png"

cv2.imwrite(str(out_cipher_path), cv2.cvtColor(cipher, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dec_path), cv2.cvtColor(decrypted, cv2.COLOR_RGB2BGR))

print("\nProses selesai. Cek folder data/ untuk melihat artefak gambar.")