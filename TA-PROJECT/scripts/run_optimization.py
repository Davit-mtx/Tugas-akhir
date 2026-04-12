
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup Path agar folder src/ terbaca
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ho.optimizer import run_ho
from src.ho.fitness import evaluate_fitness
from src.crypto.pipeline import encrypt_baseline, BaselineConfig

def main():
    print("=== SISTEM OPTIMASI PARAMETER ENKRIPSI (HO) ===")
    
    # 1. Persiapan Data (Gunakan 1 gambar sebagai sampel optimasi)
    img_path = PROJECT_ROOT / "data/samples_30_per_class/high/I04_04_01.png"
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERROR] Gambar tidak ditemukan di {img_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Kunci Hexadecimal (Statis sesuai rencana baseline)
    K_HEX = "00112233445566778899aabbccddeeff"

    # 2. Definisi Ruang Pencarian [r_min, eps, T0, Q]
    # lb = lowerbound, ub = upperbound
    lb = [3.5, 0.0001, 100, 10]    # Batas bawah parameter
    ub = [3.99, 0.05, 2000, 1000]  # Batas atas parameter

    # 3. Konfigurasi Algoritma HO
    # Untuk pengujian awal, gunakan angka kecil agar cepat selesai
    SearchAgents = 10    # Jumlah Hippo (Populasi)
    Max_iterations = 20  # Jumlah iterasi pencarian

    # Fungsi pembungkus (wrapper) agar HO hanya fokus mencari nilai X
    def objective_function(X):
        return evaluate_fitness(img_rgb, K_HEX, X)

    print(f"\n[START] Menjalankan HO pada {img_path.name}")
    print(f"Konfigurasi: {SearchAgents} Agents, {Max_iterations} Iterations")
    print("-" * 50)

    # 4. Eksekusi Optimasi
    best_score, best_pos, ho_curve = run_ho(
        SearchAgents, 
        Max_iterations, 
        lb, 
        ub, 
        objective_function
    )

    # 5. Menampilkan Hasil Terbaik
    print("-" * 50)
    print("=== HASIL OPTIMASI TERBAIK ===")
    print(f"Best Fitness Value : {best_score:.6f}")
    print(f"Optimal r_min      : {best_pos[0]:.6f}")
    print(f"Optimal epsilon    : {best_pos[1]:.6f}")
    print(f"Optimal T0 (Shift) : {int(round(best_pos[2]))}")
    print(f"Optimal Q (Iter)   : {int(round(best_pos[3]))}")

    # 6. Verifikasi Visual (Opsional: Simpan gambar hasil optimasi)
    opt_cfg = BaselineConfig(
        r_min=best_pos[0], 
        eps=best_pos[1], 
        T0=int(round(best_pos[2])), 
        Q=int(round(best_pos[3]))
    )
    cipher_opt, _ = encrypt_baseline(img_rgb, K_HEX, cfg=opt_cfg)
    
    out_path = PROJECT_ROOT / "data/results/optimized_sample.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(cipher_opt, cv2.COLOR_RGB2BGR))
    
    # 7. Plot Kurva Konvergensi
    plt.figure(figsize=(10, 5))
    plt.plot(ho_curve, linewidth=2, color='brown')
    plt.title(f"Kurva Konvergensi HO - {img_path.name}")
    plt.xlabel("Iterasi")
    plt.ylabel("Skor Fitness Terbaik")
    plt.grid(True)
    
    # Simpan plot kurva
    plot_path = PROJECT_ROOT / "data/results/convergence_curve.png"
    plt.savefig(plot_path)
    print(f"\nKurva konvergensi disimpan di: {plot_path}")
    print(f"Sampel gambar teroptimasi disimpan di: {out_path}")
    
    plt.show()

if __name__ == "__main__":
    main()