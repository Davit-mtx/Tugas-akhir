"""
Backend Tab Proses Citra untuk GUI TA.

Fungsi utama:
1. Membaca payload validasi dari InputValidationBackend.
2. Menjalankan proses baseline menggunakan X0.
3. Menjalankan proses optimized menggunakan X*(P_i).
4. Menyimpan output cipher-image dan citra dekripsi ke folder output.
5. Mengembalikan path output agar bisa ditampilkan di GUI.

Catatan:
- Backend ini tidak menampilkan messagebox.
- Backend ini hanya mengembalikan dictionary agar mudah dipakai oleh GUI.
- File pipeline.py dapat ditempatkan sebagai:
  a) src/crypto/pipeline.py, atau
  b) pipeline.py sejajar dengan main_gui.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


# =========================================================
# IMPORT PIPELINE ENKRIPSI-DEKRIPSI
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Struktur project penelitian penuh
    from src.crypto.pipeline import encrypt_baseline, decrypt_baseline, BaselineConfig
except Exception:
    try:
        # Struktur sederhana: pipeline.py sejajar dengan main_gui.py
        from pipeline import encrypt_baseline, decrypt_baseline, BaselineConfig
    except Exception as exc:
        raise ImportError(
            "Gagal import pipeline enkripsi-dekripsi. "
            "Pastikan pipeline.py berada sejajar dengan main_gui.py, "
            "atau berada pada src/crypto/pipeline.py."
        ) from exc


DEFAULT_KEY_HEX = "00112233445566778899aabbccddeeff"


class ProcessPipelineBackend:
    """Backend untuk menjalankan baseline dan optimized dari GUI."""

    def __init__(
        self,
        output_root: str | Path = "output/gui_process",
        key_hex: str = DEFAULT_KEY_HEX,
    ) -> None:
        self.output_root = Path(output_root)
        self.key_hex = key_hex
        self.last_result: Optional[Dict[str, Any]] = None

    # =====================================================
    # PUBLIC API
    # =====================================================

    def run_baseline_and_optimized(
        self,
        validation_payload: Dict[str, Any],
        output_root: str | Path | None = None,
        key_hex: str | None = None,
    ) -> Dict[str, Any]:
        """
        Menjalankan baseline dan optimized untuk satu citra valid.

        Parameters
        ----------
        validation_payload:
            Payload dari InputValidationBackend.validate_image() atau
            InputValidationBackend.get_current_payload(). Payload harus memuat:
            - ok
            - image_path
            - input_file_name
            - category
            - x0
            - xopt

        output_root:
            Folder output opsional. Jika None, menggunakan self.output_root.

        key_hex:
            Kunci 128-bit hex opsional. Jika None, menggunakan self.key_hex.

        Returns
        -------
        dict
            Hasil proses dengan path output gambar baseline dan optimized.
        """
        if not validation_payload or not validation_payload.get("ok"):
            return self._fail("Payload validasi belum valid. Input CSV dan citra valid terlebih dahulu.")

        image_path = Path(validation_payload.get("image_path", ""))
        if not image_path.exists():
            return self._fail(f"File citra tidak ditemukan: {image_path}")

        x0 = validation_payload.get("x0")
        xopt = validation_payload.get("xopt")
        if not isinstance(x0, dict) or not isinstance(xopt, dict):
            return self._fail("Payload tidak memiliki X0 atau X*(P_i) dalam format dictionary.")

        key_hex = key_hex or self.key_hex
        output_root = Path(output_root) if output_root is not None else self.output_root

        try:
            image_rgb = self.read_rgb_image(image_path)
        except Exception as exc:
            return self._fail(f"Gagal membaca citra: {exc}")

        category = str(validation_payload.get("category") or "uncategorized")
        file_name = str(validation_payload.get("input_file_name") or image_path.name)
        stem = Path(file_name).stem

        # Folder output per citra
        output_dir = output_root / category / stem
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            baseline_result = self._run_single_mode(
                mode_name="baseline",
                image_rgb=image_rgb,
                image_stem=stem,
                cfg=self.build_config_from_dict(x0),
                output_dir=output_dir,
                key_hex=key_hex,
            )

            optimized_result = self._run_single_mode(
                mode_name="optimized",
                image_rgb=image_rgb,
                image_stem=stem,
                cfg=self.build_config_from_dict(xopt),
                output_dir=output_dir,
                key_hex=key_hex,
            )

            # Simpan juga plaintext agar semua visual bersumber dari folder output yang sama
            original_path = output_dir / f"{stem}_original.png"
            self.save_rgb_image(original_path, image_rgb)

            result = {
                "ok": True,
                "message": "Proses baseline dan optimized berhasil dijalankan.",
                "key_hex": key_hex,
                "input": {
                    "image_path": str(image_path),
                    "original_path": str(original_path),
                    "file_name": file_name,
                    "category": category,
                    "best_fitness": validation_payload.get("best_fitness"),
                },
                "baseline": baseline_result,
                "optimized": optimized_result,
            }

            self.last_result = result
            return result

        except Exception as exc:
            return self._fail(f"Proses enkripsi-dekripsi gagal: {exc}")

    def get_last_result(self) -> Dict[str, Any]:
        """Mengembalikan hasil proses terakhir."""
        if self.last_result is None:
            return {"ok": False, "message": "Belum ada proses citra yang dijalankan."}
        return self.last_result

    # =====================================================
    # CONFIG DAN MODE PROCESSOR
    # =====================================================

    def build_config_from_dict(self, params: Dict[str, Any]) -> BaselineConfig:
        """
        Membentuk BaselineConfig dari dictionary parameter.

        Parameter yang diharapkan:
        {
            "r_min": float,
            "eps": float,
            "T0": int,
            "Q": int
        }
        """
        required = ["r_min", "eps", "T0", "Q"]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Parameter tidak lengkap. Kolom hilang: {missing}")

        return BaselineConfig(
            r_min=float(params["r_min"]),
            eps=float(params["eps"]),
            T0=int(round(float(params["T0"]))),
            Q=int(round(float(params["Q"]))),
        )

    def _run_single_mode(
        self,
        mode_name: str,
        image_rgb: np.ndarray,
        image_stem: str,
        cfg: BaselineConfig,
        output_dir: Path,
        key_hex: str,
    ) -> Dict[str, Any]:
        """Menjalankan satu mode: baseline atau optimized."""
        start_enc = time.time()
        cipher_rgb, _ = encrypt_baseline(image_rgb, key_hex, cfg=cfg, return_debug=False)
        enc_time = time.time() - start_enc

        start_dec = time.time()
        decrypted_rgb = decrypt_baseline(cipher_rgb, key_hex, cfg=cfg)
        dec_time = time.time() - start_dec

        lossless = bool(np.array_equal(image_rgb, decrypted_rgb))

        cipher_path = output_dir / f"{image_stem}_{mode_name}_cipher.png"
        decrypted_path = output_dir / f"{image_stem}_{mode_name}_decrypted.png"

        self.save_rgb_image(cipher_path, cipher_rgb)
        self.save_rgb_image(decrypted_path, decrypted_rgb)

        return {
            "mode": mode_name,
            "cipher_path": str(cipher_path),
            "decrypted_path": str(decrypted_path),
            "enc_time": round(enc_time, 6),
            "dec_time": round(dec_time, 6),
            "lossless": lossless,
            "config": {
                "r_min": float(cfg.r_min),
                "eps": float(cfg.eps),
                "T0": int(cfg.T0),
                "Q": int(cfg.Q),
                "r_max": float(cfg.r_max),
            },
        }

    # =====================================================
    # IMAGE I/O
    # =====================================================

    def read_rgb_image(self, image_path: str | Path) -> np.ndarray:
        """Membaca citra dengan OpenCV lalu mengubah BGR menjadi RGB."""
        image_path = Path(image_path)
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Gambar tidak dapat dibaca: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def save_rgb_image(self, out_path: str | Path, image_rgb: np.ndarray) -> None:
        """Menyimpan citra RGB sebagai file gambar melalui konversi ke BGR."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image_rgb = np.asarray(image_rgb)
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(out_path), image_bgr)
        if not ok:
            raise IOError(f"Gagal menyimpan gambar: {out_path}")

    # =====================================================
    # INTERNAL HELPER
    # =====================================================

    def _fail(self, message: str) -> Dict[str, Any]:
        result = {"ok": False, "message": message}
        self.last_result = result
        return result
