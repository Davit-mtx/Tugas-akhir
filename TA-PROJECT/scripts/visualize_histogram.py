
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    print("=== MENGHASILKAN GRAFIK HISTOGRAM RGB (Subbab 2.8.1) ===")
    
    # Ambil salah satu gambar Juara dari folder appendix
    # PASTIKAN nama file ini ada di dalam folder appendix Anda!
    out_dir = PROJECT_ROOT / "data/results/plots/analisis_histogram"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    appendix_dir = PROJECT_ROOT / "data/results/appendix"
    
    # Mencari gambar Original dan Cipher dari salah satu juara (misal kategori High)
    img_orig_path = list(appendix_dir.glob("*_1_Original.png"))[0]
    img_ciph_path = list(appendix_dir.glob("*_2_Cipher_K1.png"))[0]

    img_orig = cv2.imread(str(img_orig_path))
    img_ciph = cv2.imread(str(img_ciph_path))

    colors = ('b', 'g', 'r')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Analisis Histogram RGB: Plaintext vs Cipher-image', fontsize=16)

    for i, col in enumerate(colors):
        # Histogram Original
        hist_orig = cv2.calcHist([img_orig], [i], None, [256], [0, 256])
        axes[0, i].plot(hist_orig, color=col)
        axes[0, i].set_title(f'Plaintext - Kanal {col.upper()}')
        axes[0, i].set_xlim([0, 256])
        
        # Histogram Cipher
        hist_ciph = cv2.calcHist([img_ciph], [i], None, [256], [0, 256])
        axes[1, i].plot(hist_ciph, color=col)
        axes[1, i].set_title(f'Cipher-image - Kanal {col.upper()}')
        axes[1, i].set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(out_dir / "Histogram_Analysis.png", dpi=300)
    print(f"Berhasil disimpan di: {out_dir}/Histogram_Analysis.png")

if __name__ == "__main__":
    main()