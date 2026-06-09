"""
Contoh pengujian backend Tab Input & Validasi tanpa GUI.

Cara pakai:
1. Simpan folder backend sejajar dengan file ini atau jalankan dari root project.
2. Ubah CSV_PATH dan IMAGE_PATH sesuai lokasi file di laptop kamu.
3. Jalankan:
   python examples/test_input_validation.py
"""

from pathlib import Path
import sys

# Agar contoh bisa dijalankan dari folder examples maupun root project
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import InputValidationBackend


CSV_PATH = "summary_optimized_results.csv"
IMAGE_PATH = "I04_04_01.png"


if __name__ == "__main__":
    backend = InputValidationBackend()

    csv_result = backend.load_csv(CSV_PATH)
    print("=== HASIL LOAD CSV ===")
    print(csv_result)
    print()

    image_result = backend.validate_image(IMAGE_PATH)
    print("=== HASIL VALIDASI CITRA ===")
    print(backend.format_validation_summary_text(image_result))
    print()

    if image_result["ok"]:
        print("=== X0 BASELINE ===")
        print(backend.format_parameter_text("X0 Baseline", image_result["x0"]))
        print()

        print("=== X*(P_i) OPTIMIZED ===")
        print(backend.format_parameter_text("X*(P_i) Optimized", image_result["xopt"]))
