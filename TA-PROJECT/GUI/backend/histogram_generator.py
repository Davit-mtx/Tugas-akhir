"""
Backend Tab Histogram untuk GUI TA.

Fungsi utama:
1. Mengambil hasil proses dari ProcessPipelineBackend.
2. Membuat histogram RGB untuk baseline plaintext, baseline cipher,
   optimized plaintext, dan optimized cipher.
3. Menyimpan histogram sebagai PNG dan mengembalikan path ke GUI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHANNELS_RGB = [
    ("R", 0, "red"),
    ("G", 1, "green"),
    ("B", 2, "blue"),
]


class HistogramGeneratorBackend:
    """Backend untuk membuat gambar histogram RGB dari hasil proses GUI."""

    def __init__(self, output_root: str | Path = "output/gui_histogram") -> None:
        self.output_root = Path(output_root)
        self.last_result: Optional[Dict[str, Any]] = None

    # =====================================================
    # PUBLIC API
    # =====================================================

    def generate_from_process_result(self, process_result: Dict[str, Any]) -> Dict[str, Any]:
        if not process_result or not process_result.get("ok"):
            return self._fail("Belum ada hasil proses citra yang valid. Jalankan baseline dan optimized terlebih dahulu.")

        try:
            input_info = process_result.get("input", {})
            category = str(input_info.get("category") or "uncategorized")
            file_name = str(input_info.get("file_name") or Path(input_info.get("original_path", "image")).name)
            stem = Path(file_name).stem

            out_dir = self.output_root / category / stem
            out_dir.mkdir(parents=True, exist_ok=True)

            original_path = input_info["original_path"]
            baseline_cipher_path = process_result["baseline"]["cipher_path"]
            optimized_cipher_path = process_result["optimized"]["cipher_path"]

            outputs = {
                "baseline_plaintext": str(out_dir / f"{stem}_hist_baseline_plaintext.png"),
                "baseline_cipher": str(out_dir / f"{stem}_hist_baseline_cipher.png"),
                "optimized_plaintext": str(out_dir / f"{stem}_hist_optimized_plaintext.png"),
                "optimized_cipher": str(out_dir / f"{stem}_hist_optimized_cipher.png"),
            }

            self.plot_single_histogram(original_path, outputs["baseline_plaintext"], "Baseline - Plaintext")
            self.plot_single_histogram(baseline_cipher_path, outputs["baseline_cipher"], "Baseline - Cipher Image")
            self.plot_single_histogram(original_path, outputs["optimized_plaintext"], "Optimized - Plaintext")
            self.plot_single_histogram(optimized_cipher_path, outputs["optimized_cipher"], "Optimized - Cipher Image")

            result = {
                "ok": True,
                "message": "Histogram RGB baseline dan optimized berhasil dibuat.",
                "input": input_info,
                "histogram_paths": outputs,
            }
            self.last_result = result
            return result
        except Exception as exc:
            return self._fail(f"Pembuatan histogram gagal: {exc}")

    def get_last_result(self) -> Dict[str, Any]:
        if self.last_result is None:
            return {"ok": False, "message": "Belum ada histogram yang dibuat."}
        return self.last_result

    # =====================================================
    # PLOTTING
    # =====================================================

    def plot_single_histogram(self, image_path: str | Path, out_path: str | Path, title: str) -> None:
        img_rgb = self.read_rgb_image(image_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 4.5))
        for channel_name, channel_idx, color_name in CHANNELS_RGB:
            hist = cv2.calcHist([img_rgb], [channel_idx], None, [256], [0, 256])
            plt.plot(hist, color=color_name, linewidth=1.2, label=f"Kanal {channel_name}")

        plt.title(title)
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Frekuensi")
        plt.xlim([0, 256])
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()

    def read_rgb_image(self, image_path: str | Path) -> np.ndarray:
        image_path = Path(image_path)
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Gambar tidak dapat dibaca: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _fail(self, message: str) -> Dict[str, Any]:
        result = {"ok": False, "message": message}
        self.last_result = result
        return result
