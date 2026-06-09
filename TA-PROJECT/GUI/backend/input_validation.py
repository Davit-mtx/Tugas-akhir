"""
Backend Tab Input & Validasi untuk GUI TA.

Fungsi utama:
1. Membaca file CSV hasil optimized.
2. Memastikan struktur CSV sesuai kebutuhan penelitian.
3. Menjadikan CSV sebagai database validasi citra.
4. Memvalidasi citra input berdasarkan kolom "File Name".
5. Mengambil variabel keputusan X0 dan X*(P_i).

Catatan:
- Backend ini tidak menampilkan messagebox.
- Backend ini hanya mengembalikan dictionary agar mudah dipakai oleh GUI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import (
    ALLOWED_IMAGE_EXTENSIONS,
    BASELINE_X0,
    BEST_FITNESS_COLUMN,
    CATEGORY_COLUMN,
    FILENAME_COLUMN,
    OPTIMIZED_PARAMETER_COLUMNS,
)


class InputValidationBackend:
    """Backend untuk membaca CSV optimized dan memvalidasi citra input."""

    def __init__(self) -> None:
        self.csv_path: Optional[Path] = None
        self.df: Optional[pd.DataFrame] = None
        self.selected_image_path: Optional[Path] = None
        self.selected_row: Optional[pd.Series] = None
        self.last_validation_result: Optional[Dict[str, Any]] = None

    # =========================================================
    # CSV LOADING
    # =========================================================

    def load_csv(self, csv_path: str | Path) -> Dict[str, Any]:
        """
        Membaca CSV optimized dan memvalidasi kolom wajib.

        Parameters
        ----------
        csv_path:
            Path menuju CSV hasil optimized.

        Returns
        -------
        dict
            Hasil pembacaan CSV dengan format:
            {
                "ok": bool,
                "message": str,
                "csv_path": str | None,
                "n_rows": int,
                "n_columns": int,
                "columns": list[str],
                "category_counts": dict,
                "missing_columns": list[str],
            }
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            return self._fail(f"File CSV tidak ditemukan: {csv_path}")

        if csv_path.suffix.lower() != ".csv":
            return self._fail("File yang dipilih bukan CSV.")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            return self._fail(f"CSV gagal dibaca: {exc}")

        if df.empty:
            return self._fail("CSV berhasil dibaca, tetapi isinya kosong.")

        missing_columns = self.get_missing_required_columns(df)
        if missing_columns:
            return {
                "ok": False,
                "message": "CSV tidak memiliki kolom wajib yang dibutuhkan.",
                "csv_path": str(csv_path),
                "n_rows": int(len(df)),
                "n_columns": int(len(df.columns)),
                "columns": list(df.columns),
                "category_counts": self._get_category_counts(df),
                "missing_columns": missing_columns,
            }

        # Simpan state jika valid
        self.csv_path = csv_path
        self.df = df
        self.selected_image_path = None
        self.selected_row = None
        self.last_validation_result = None

        return {
            "ok": True,
            "message": "CSV optimized berhasil dibaca dan strukturnya valid.",
            "csv_path": str(csv_path),
            "file_name": csv_path.name,
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "category_counts": self._get_category_counts(df),
            "missing_columns": [],
        }

    def get_missing_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Mengembalikan daftar kolom wajib yang tidak ada pada CSV."""
        required_columns = [
            FILENAME_COLUMN,
            CATEGORY_COLUMN,
            BEST_FITNESS_COLUMN,
            *OPTIMIZED_PARAMETER_COLUMNS.values(),
        ]
        return [col for col in required_columns if col not in df.columns]

    # =========================================================
    # IMAGE VALIDATION
    # =========================================================

    def validate_image(self, image_path: str | Path) -> Dict[str, Any]:
        """
        Memvalidasi citra berdasarkan database CSV optimized.

        Citra valid jika nama file atau stem nama file ditemukan di kolom File Name.

        Parameters
        ----------
        image_path:
            Path citra input.

        Returns
        -------
        dict
            Hasil validasi citra dengan format utama:
            {
                "ok": bool,
                "message": str,
                "image_path": str | None,
                "input_file_name": str,
                "matched_file_name": str | None,
                "match_type": "exact" | "stem" | None,
                "row_index": int | None,
                "category": str | None,
                "best_fitness": float | None,
                "x0": dict | None,
                "xopt": dict | None,
                "row_data": dict | None,
            }
        """
        if self.df is None:
            return self._validation_fail(
                image_path=image_path,
                message="CSV optimized belum dimasukkan. Silakan input CSV terlebih dahulu.",
            )

        image_path = Path(image_path)

        if not image_path.exists():
            return self._validation_fail(
                image_path=image_path,
                message=f"File citra tidak ditemukan: {image_path}",
            )

        if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            return self._validation_fail(
                image_path=image_path,
                message=f"Ekstensi citra tidak didukung: {image_path.suffix}",
            )

        input_file_name = image_path.name.strip()
        input_stem = image_path.stem.strip()

        df = self.df.copy()
        db_file_names = df[FILENAME_COLUMN].astype(str).str.strip().apply(os.path.basename)
        db_stems = db_file_names.apply(lambda name: Path(name).stem)

        # Pencocokan lapis 1: nama file penuh, case-insensitive
        exact_mask = db_file_names.str.lower() == input_file_name.lower()
        exact_matches = df[exact_mask]

        if not exact_matches.empty:
            row = exact_matches.iloc[0]
            row_index = int(exact_matches.index[0])
            result = self._build_success_result(
                image_path=image_path,
                row=row,
                row_index=row_index,
                matched_file_name=str(db_file_names.loc[row_index]),
                match_type="exact",
            )
            self._store_validation_success(image_path, row, result)
            return result

        # Pencocokan lapis 2: nama file tanpa ekstensi, case-insensitive
        stem_mask = db_stems.str.lower() == input_stem.lower()
        stem_matches = df[stem_mask]

        if not stem_matches.empty:
            row = stem_matches.iloc[0]
            row_index = int(stem_matches.index[0])
            result = self._build_success_result(
                image_path=image_path,
                row=row,
                row_index=row_index,
                matched_file_name=str(db_file_names.loc[row_index]),
                match_type="stem",
            )
            self._store_validation_success(image_path, row, result)
            return result

        result = self._validation_fail(
            image_path=image_path,
            message="Citra tidak ditemukan dalam database CSV optimized.",
        )
        self.selected_image_path = None
        self.selected_row = None
        self.last_validation_result = result
        return result

    # =========================================================
    # PARAMETER EXTRACTION
    # =========================================================

    def get_baseline_x0(self) -> Dict[str, Any]:
        """Mengembalikan variabel keputusan baseline X0."""
        return dict(BASELINE_X0)

    def extract_xopt_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """Mengambil X*(P_i) dari satu baris CSV."""
        return {
            param_name: self._to_python_scalar(row[column_name])
            for param_name, column_name in OPTIMIZED_PARAMETER_COLUMNS.items()
        }

    def get_current_payload(self) -> Dict[str, Any]:
        """
        Mengembalikan payload terakhir yang siap digunakan tab Proses Citra.

        Fungsi ini berguna setelah validate_image() berhasil.
        """
        if self.last_validation_result is None:
            return {
                "ok": False,
                "message": "Belum ada citra yang divalidasi.",
            }

        return self.last_validation_result

    # =========================================================
    # DISPLAY FORMATTER UNTUK GUI
    # =========================================================

    def format_parameter_text(self, title: str, parameters: Dict[str, Any]) -> str:
        """Membentuk teks parameter agar mudah ditampilkan di CTkTextbox."""
        lines = [title, ""]
        for key, value in parameters.items():
            lines.append(f"{key:<8}: {value}")
        return "\n".join(lines)

    def format_validation_summary_text(self, result: Dict[str, Any]) -> str:
        """Membentuk teks ringkasan validasi untuk panel GUI."""
        if not result.get("ok"):
            return f"Status  : GAGAL\nPesan   : {result.get('message', '-')}"

        return (
            f"Status        : VALID\n"
            f"Input File    : {result.get('input_file_name', '-')}\n"
            f"Matched File  : {result.get('matched_file_name', '-')}\n"
            f"Match Type    : {result.get('match_type', '-')}\n"
            f"Kategori      : {result.get('category', '-')}\n"
            f"Best Fitness  : {result.get('best_fitness', '-')}"
        )

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _build_success_result(
        self,
        image_path: Path,
        row: pd.Series,
        row_index: int,
        matched_file_name: str,
        match_type: str,
    ) -> Dict[str, Any]:
        category = row.get(CATEGORY_COLUMN, None)
        best_fitness = row.get(BEST_FITNESS_COLUMN, None)

        return {
            "ok": True,
            "message": "Citra valid dan ditemukan dalam CSV optimized.",
            "image_path": str(image_path),
            "input_file_name": image_path.name,
            "matched_file_name": matched_file_name,
            "match_type": match_type,
            "row_index": row_index,
            "category": self._to_python_scalar(category),
            "best_fitness": self._to_python_scalar(best_fitness),
            "x0": self.get_baseline_x0(),
            "xopt": self.extract_xopt_from_row(row),
            "row_data": {str(k): self._to_python_scalar(v) for k, v in row.to_dict().items()},
        }

    def _store_validation_success(
        self,
        image_path: Path,
        row: pd.Series,
        result: Dict[str, Any],
    ) -> None:
        self.selected_image_path = image_path
        self.selected_row = row
        self.last_validation_result = result

    def _validation_fail(self, image_path: str | Path, message: str) -> Dict[str, Any]:
        image_path = Path(image_path)
        return {
            "ok": False,
            "message": message,
            "image_path": str(image_path),
            "input_file_name": image_path.name,
            "matched_file_name": None,
            "match_type": None,
            "row_index": None,
            "category": None,
            "best_fitness": None,
            "x0": None,
            "xopt": None,
            "row_data": None,
        }

    def _fail(self, message: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "message": message,
            "csv_path": None,
            "n_rows": 0,
            "n_columns": 0,
            "columns": [],
            "category_counts": {},
            "missing_columns": [],
        }

    def _get_category_counts(self, df: pd.DataFrame) -> Dict[str, int]:
        if CATEGORY_COLUMN not in df.columns:
            return {}
        counts = df[CATEGORY_COLUMN].value_counts(dropna=False).to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def _to_python_scalar(self, value: Any) -> Any:
        """Mengubah tipe numpy/pandas menjadi tipe Python biasa agar mudah dipakai GUI."""
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value
