
from pathlib import Path
import cv2
import numpy as np

from src.crypto.pipeline import encrypt_baseline, BaselineConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------
# 1. Pilih 1 gambar
# -----------------------------------
img_path = PROJECT_ROOT / "data/samples_30_per_class/high/I04_04_01.png"  # ganti sesuai file kamu

# -----------------------------------
# 2. Baca gambar
# -----------------------------------
img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
if img_bgr is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan atau gagal dibaca: {img_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# -----------------------------------
# 3. Siapkan key dan config baseline
# -----------------------------------
K = "00112233445566778899aabbccddeeff"
cfg = BaselineConfig(
    r_min=3.70,
    eps=0.002,
    T0=500,   # biarkan dulu, karena plaintext-related sudah pakai transient Oravec khusus
    Q=256
)

# -----------------------------------
# 4. Jalankan enkripsi
# -----------------------------------
cipher, debug = encrypt_baseline(img_rgb, K, cfg=cfg, return_debug=True)

# -----------------------------------
# 5. Print info penting
# -----------------------------------
print("=== SMOKE TEST BASELINE ENCRYPTION ===")
print("Input shape      :", img_rgb.shape, img_rgb.dtype)
print("Cipher shape     :", cipher.shape, cipher.dtype)

for name in ["Pprime", "Ppr", "Pconf", "Pdiff", "Pout"]: 
    if name in debug:
        arr = debug[name]
        print(f"{name:10s}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

# -----------------------------------
# 6. Simpan hasil cipher untuk cek visual
# -----------------------------------
out_path = Path("data") / "smoke_cipher.png"
cipher_bgr = cv2.cvtColor(cipher, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(out_path), cipher_bgr)

print(f"Hasil cipher disimpan di: {out_path}")
print("Smoke test selesai.")