"""
Backend Tab Evaluasi Metrik untuk GUI TA.

Fungsi utama:
1. Mengambil hasil proses dari ProcessPipelineBackend.
2. Menghitung metrik RGB untuk baseline dan optimized.
3. Menghasilkan teks tabel yang siap ditampilkan di CTkTextbox.

Catatan:
- Backend ini tidak menampilkan messagebox.
- NPCR/UACI dihitung dengan plaintext yang dimodifikasi 1 bit pada piksel [0,0,0]
  kanal R, sesuai pola evaluasi pada script penelitian.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


# =========================================================
# IMPORT PIPELINE DAN METRIK
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.crypto.pipeline import encrypt_baseline, BaselineConfig
except Exception:
    try:
        from pipeline import encrypt_baseline, BaselineConfig
    except Exception as exc:
        raise ImportError(
            "Gagal import pipeline. Pastikan pipeline.py berada sejajar dengan main_gui.py, "
            "atau berada pada src/crypto/pipeline.py."
        ) from exc

try:
    from src.metrics.metric import calculate_entropy, calculate_correlation, calculate_npcr_uaci
except Exception:
    try:
        from metric import calculate_entropy, calculate_correlation, calculate_npcr_uaci
    except Exception as exc:
        raise ImportError(
            "Gagal import metric. Pastikan metric.py berada sejajar dengan main_gui.py, "
            "atau berada pada src/metrics/metric.py."
        ) from exc


CHANNELS_RGB = {
    "R": 0,
    "G": 1,
    "B": 2,
}


class EvaluationMetricsBackend:
    """Backend untuk menghitung dan memformat evaluasi metrik baseline dan optimized."""

    def __init__(self) -> None:
        self.last_result: Optional[Dict[str, Any]] = None

    # =====================================================
    # PUBLIC API
    # =====================================================

    def evaluate_process_result(self, process_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Menghitung metrik baseline dan optimized dari hasil proses citra.

        process_result diharapkan berasal dari:
            ProcessPipelineBackend.run_baseline_and_optimized(...)
        """
        if not process_result or not process_result.get("ok"):
            return self._fail("Belum ada hasil proses citra yang valid. Jalankan baseline dan optimized terlebih dahulu.")

        try:
            original_path = process_result["input"]["original_path"]
            original_rgb = self.read_rgb_image(original_path)
            key_hex = process_result.get("key_hex", "00112233445566778899aabbccddeeff")

            baseline_metrics = self._evaluate_mode(
                original_rgb=original_rgb,
                cipher_path=process_result["baseline"]["cipher_path"],
                mode_result=process_result["baseline"],
                key_hex=key_hex,
            )

            optimized_metrics = self._evaluate_mode(
                original_rgb=original_rgb,
                cipher_path=process_result["optimized"]["cipher_path"],
                mode_result=process_result["optimized"],
                key_hex=key_hex,
            )

            result = {
                "ok": True,
                "message": "Evaluasi metrik baseline dan optimized berhasil dihitung.",
                "input": process_result.get("input", {}),
                "baseline": baseline_metrics,
                "optimized": optimized_metrics,
                "baseline_table": self.format_metric_table("Evaluasi Metrik - Baseline", baseline_metrics),
                "optimized_table": self.format_metric_table("Evaluasi Metrik - Optimized", optimized_metrics),
            }
            self.last_result = result
            return result
        except Exception as exc:
            return self._fail(f"Evaluasi metrik gagal: {exc}")

    def get_last_result(self) -> Dict[str, Any]:
        if self.last_result is None:
            return {"ok": False, "message": "Belum ada evaluasi metrik yang dijalankan."}
        return self.last_result

    # =====================================================
    # CORE EVALUATION
    # =====================================================

    def _evaluate_mode(
        self,
        original_rgb: np.ndarray,
        cipher_path: str | Path,
        mode_result: Dict[str, Any],
        key_hex: str,
    ) -> Dict[str, Any]:
        cipher_rgb = self.read_rgb_image(cipher_path)

        # Plaintext dimodifikasi 1 bit pada piksel [0,0,0] kanal R.
        modified_rgb = original_rgb.copy()
        modified_rgb[0, 0, 0] = int(modified_rgb[0, 0, 0]) ^ 1

        cfg = self.build_config_from_dict(mode_result.get("config", {}))
        cipher_modified_rgb, _ = encrypt_baseline(modified_rgb, key_hex, cfg=cfg, return_debug=False)

        metrics = self.calculate_rgb_metrics(cipher_rgb, cipher_modified_rgb)
        metrics["Enc_Time (s)"] = float(mode_result.get("enc_time", 0.0))
        metrics["Dec_Time (s)"] = float(mode_result.get("dec_time", 0.0))
        metrics["Lossless"] = bool(mode_result.get("lossless", False))
        metrics["Config"] = {
            "r_min": float(cfg.r_min),
            "eps": float(cfg.eps),
            "T0": int(cfg.T0),
            "Q": int(cfg.Q),
            "r_max": float(cfg.r_max),
        }
        return metrics

    def calculate_rgb_metrics(self, cipher_original: np.ndarray, cipher_modified: np.ndarray) -> Dict[str, Any]:
        if cipher_original.ndim != 3 or cipher_original.shape[2] != 3:
            raise ValueError("Evaluasi RGB membutuhkan cipher-image dengan format H x W x 3.")

        entropy_values = []
        corr_h_values = []
        corr_v_values = []
        corr_d_values = []
        npcr_values = []
        uaci_values = []

        metrics: Dict[str, Any] = {
            "Entropy": {},
            "Corr Horizontal": {},
            "Corr Vertical": {},
            "Corr Diagonal": {},
            "NPCR (%)": {},
            "UACI (%)": {},
        }

        for channel_name, channel_idx in CHANNELS_RGB.items():
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

            metrics["Entropy"][channel_name] = round(float(entropy_ch), 5)
            metrics["Corr Horizontal"][channel_name] = round(float(corr_ch["horizontal"]), 5)
            metrics["Corr Vertical"][channel_name] = round(float(corr_ch["vertical"]), 5)
            metrics["Corr Diagonal"][channel_name] = round(float(corr_ch["diagonal"]), 5)
            metrics["NPCR (%)"][channel_name] = round(float(npcr_ch), 5)
            metrics["UACI (%)"][channel_name] = round(float(uaci_ch), 5)

        metrics["Entropy"]["Average"] = round(float(np.mean(entropy_values)), 5)
        metrics["Corr Horizontal"]["Average"] = round(float(np.mean(corr_h_values)), 5)
        metrics["Corr Vertical"]["Average"] = round(float(np.mean(corr_v_values)), 5)
        metrics["Corr Diagonal"]["Average"] = round(float(np.mean(corr_d_values)), 5)
        metrics["NPCR (%)"]["Average"] = round(float(np.mean(npcr_values)), 5)
        metrics["UACI (%)"]["Average"] = round(float(np.mean(uaci_values)), 5)

        return metrics

    # =====================================================
    # FORMATTER UNTUK GUI
    # =====================================================

    def format_metric_table(self, title: str, metrics: Dict[str, Any]) -> str:
        rows = [
            "Metrik              R          G          B       Average",
            "----------------------------------------------------------",
        ]

        for metric_name in [
            "Entropy",
            "Corr Horizontal",
            "Corr Vertical",
            "Corr Diagonal",
            "NPCR (%)",
            "UACI (%)",
        ]:
            values = metrics.get(metric_name, {})
            rows.append(
                f"{metric_name:<18}"
                f"{self._fmt(values.get('R')):>10}"
                f"{self._fmt(values.get('G')):>10}"
                f"{self._fmt(values.get('B')):>10}"
                f"{self._fmt(values.get('Average')):>10}"
            )

        rows.append("----------------------------------------------------------")
        rows.append(f"{'Waktu Enkripsi':<18}{'-':>10}{'-':>10}{'-':>10}{self._fmt(metrics.get('Enc_Time (s)'), suffix=' s'):>10}")
        rows.append(f"{'Waktu Dekripsi':<18}{'-':>10}{'-':>10}{'-':>10}{self._fmt(metrics.get('Dec_Time (s)'), suffix=' s'):>10}")
        rows.append(f"{'Lossless':<18}{'-':>10}{'-':>10}{'-':>10}{str(metrics.get('Lossless')):>10}")

        cfg = metrics.get("Config", {})
        rows.append("\nParameter:")
        rows.append(f"r_min = {cfg.get('r_min', '-')} | eps = {cfg.get('eps', '-')} | T0 = {cfg.get('T0', '-')} | Q = {cfg.get('Q', '-')}")

        return "\n".join(rows)

    def _fmt(self, value: Any, suffix: str = "") -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.5f}{suffix}"
        return str(value)

    # =====================================================
    # IMAGE, CONFIG, HELPER
    # =====================================================

    def read_rgb_image(self, image_path: str | Path) -> np.ndarray:
        image_path = Path(image_path)
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Gambar tidak dapat dibaca: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def build_config_from_dict(self, params: Dict[str, Any]) -> BaselineConfig:
        required = ["r_min", "eps", "T0", "Q"]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Parameter config tidak lengkap: {missing}")

        return BaselineConfig(
            r_min=float(params["r_min"]),
            eps=float(params["eps"]),
            T0=int(round(float(params["T0"]))),
            Q=int(round(float(params["Q"]))),
        )

    def _fail(self, message: str) -> Dict[str, Any]:
        result = {"ok": False, "message": message}
        self.last_result = result
        return result
