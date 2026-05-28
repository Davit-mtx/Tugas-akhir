from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHANNELS_RGB = [
    ("R", 0, "red"),
    ("G", 1, "green"),
    ("B", 2, "blue"),
]


def read_rgb_image(image_path):
    """cv2.imread membaca BGR, sehingga harus dikonversi ke RGB agar urutan kanal sesuai penelitian."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Gambar tidak dapat dibaca: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def plot_histogram_pair(img_orig_path, img_ciph_path, out_path):
    img_orig = read_rgb_image(img_orig_path)
    img_ciph = read_rgb_image(img_ciph_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Analisis Histogram RGB: Plaintext vs Cipher-image\n{img_orig_path.name}",
        fontsize=16
    )

    for i, (channel_name, channel_idx, color_name) in enumerate(CHANNELS_RGB):
        hist_orig = cv2.calcHist([img_orig], [channel_idx], None, [256], [0, 256])
        axes[0, i].plot(hist_orig, color=color_name)
        axes[0, i].set_title(f"Plaintext - Kanal {channel_name}")
        axes[0, i].set_xlim([0, 256])
        axes[0, i].set_xlabel("Intensitas Piksel")
        axes[0, i].set_ylabel("Frekuensi")

        hist_ciph = cv2.calcHist([img_ciph], [channel_idx], None, [256], [0, 256])
        axes[1, i].plot(hist_ciph, color=color_name)
        axes[1, i].set_title(f"Cipher-image - Kanal {channel_name}")
        axes[1, i].set_xlim([0, 256])
        axes[1, i].set_xlabel("Intensitas Piksel")
        axes[1, i].set_ylabel("Frekuensi")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    print("=== MENGHASILKAN GRAFIK HISTOGRAM RGB (R, G, B) ===")

    out_dir = PROJECT_ROOT / "data/results/plots/analisis_histogram"
    out_dir.mkdir(parents=True, exist_ok=True)

    appendix_dir = PROJECT_ROOT / "data/results/appendix"
    original_files = sorted(appendix_dir.glob("*_1_Original.png"))

    if not original_files:
        print("[ERROR] Tidak ditemukan file *_1_Original.png di data/results/appendix/.")
        print("Jalankan Key_sensitivity.py terlebih dahulu untuk menghasilkan gambar appendix.")
        return

    total_saved = 0
    for img_orig_path in original_files:
        pair_prefix = img_orig_path.name.replace("_1_Original.png", "")
        img_ciph_path = appendix_dir / f"{pair_prefix}_2_Cipher_K1.png"

        if not img_ciph_path.exists():
            print(f"[SKIP] Pasangan cipher tidak ditemukan untuk: {img_orig_path.name}")
            continue

        out_path = out_dir / f"Histogram_RGB_{pair_prefix}.png"
        plot_histogram_pair(img_orig_path, img_ciph_path, out_path)
        total_saved += 1
        print(f"-> Berhasil disimpan: {out_path.name}")

    print(f"\n[SELESAI] Total histogram RGB tersimpan: {total_saved} file di {out_dir}")


if __name__ == "__main__":
    main()
