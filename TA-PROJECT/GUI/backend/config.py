"""
Konfigurasi tetap untuk backend Tab Input & Validasi GUI TA.

File ini menyimpan nama kolom CSV dan variabel keputusan baseline X0
sesuai rancangan penelitian.
"""

# Kolom utama pada CSV summary_optimized_results.csv
FILENAME_COLUMN = "File Name"
CATEGORY_COLUMN = "Category"
BEST_FITNESS_COLUMN = "Best_Fitness"

# Kolom parameter hasil optimasi X*(P_i)
OPTIMIZED_PARAMETER_COLUMNS = {
    "r_min": "Opt_r_min",
    "eps": "Opt_eps",
    "T0": "Opt_T0",
    "Q": "Opt_Q",
}

# Variabel keputusan baseline X0 berdasarkan penelitian
# X0 = [3.70, 0.02, 500, 256]
BASELINE_X0 = {
    "r_min": 3.70,
    "eps": 0.02,
    "T0": 500,
    "Q": 256,
}

# Ekstensi citra yang diizinkan
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
