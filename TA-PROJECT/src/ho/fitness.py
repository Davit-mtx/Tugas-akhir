
import numpy as np
from src.crypto.pipeline import encrypt_baseline, decrypt_baseline, BaselineConfig
from src.metrics.metric import calculate_entropy, calculate_correlation, verify_lossless

def evaluate_fitness(P: np.ndarray, K_hex: str, X_candidate: np.ndarray) -> float:
    """
    Fungsi Objektif (Fitness) sesuai Persamaan 2.17 - 2.20 di proposal.
    Mengevaluasi satu kandidat X = [r_min, eps, T0, Q].
    Target: Minimasi (Semakin mendekati 0, semakin baik).
    """
    r_min, eps, T0, Q = X_candidate
    
    # 1. Evaluasi Kendala Kelayakan Parameter (Penalty_feas)
    r_max = 4.0 - eps
    Penalty_feas = 0.0
    
    # Syarat kelayakan chaos interval: r_min < r_max
    if r_min >= r_max:
        Penalty_feas = 1e10  # M_feas (Penalti masif)
        return Penalty_feas  # Langsung kembalikan penalti untuk menghemat komputasi

    # 2. Setup Parameter Kandidat
    # Pastikan T0 dan Q dibulatkan (direparasi) ke bentuk integer
    cfg = BaselineConfig(r_min=float(r_min), eps=float(eps), T0=int(round(T0)), Q=int(round(Q)))
    
    # 3. Uji Coba Enkripsi dan Dekripsi
    try:
        C, _ = encrypt_baseline(P, K_hex, cfg=cfg, return_debug=False)
        P_hat = decrypt_baseline(C, K_hex, cfg=cfg)
    except Exception:
        # Jika terjadi error operasi matriks karena parameter ekstrem
        return 1e10

    # 4. Evaluasi Kendala Lossless (Penalty_dec)
    Penalty_dec = 0.0
    if not verify_lossless(P, P_hat):
        Penalty_dec = 1e10  # M_dec (Penalti masif kegagalan dekripsi)
        return Penalty_dec  # Hentikan evaluasi, solusi ini tidak sah

    # 5. Perhitungan Metrik Statistik
    # Menggunakan channel merah (R) sebagai sampel representatif efisiensi HO
    R_channel = C[:, :, 0]
    
    entropy = calculate_entropy(R_channel)
    corr = calculate_correlation(R_channel)
    
    # Normalisasi agar setara (Persamaan 2.15 dan 2.16)
    H_tilde = entropy / 8.0
    rho_tilde = (abs(corr['horizontal']) + abs(corr['vertical']) + abs(corr['diagonal'])) / 3.0
    
    # Pembobotan (Sama rata 50:50)
    w_H = 0.5
    w_rho = 0.5
    
    # 6. Kalkulasi Fungsi Objektif Komposit (Persamaan 2.17)
    # Penalti adalah 0 karena kandidat lolos semua pengecekan di atas
    f_X = w_H * (1.0 - H_tilde) + w_rho * rho_tilde
    
    return float(f_X)