import numpy as np
from src.crypto.pipeline import encrypt_baseline, decrypt_baseline, BaselineConfig
from src.metrics.metric import calculate_entropy, calculate_correlation, verify_lossless


def _calculate_rgb_fitness_components(C: np.ndarray) -> tuple[float, float]:
    """
    Menghitung komponen fitness berbasis RGB:
    1. Entropi ternormalisasi rata-rata RGB.
    2. Korelasi absolut rata-rata dari arah horizontal, vertikal, diagonal
       pada kanal R, G, dan B.

    Output:
        H_tilde   : rata-rata entropy/8 dari kanal R, G, B
        rho_tilde : rata-rata |korelasi| dari 3 arah x 3 kanal
    """
    if C.ndim != 3 or C.shape[2] != 3:
        raise ValueError("Fitness RGB membutuhkan citra dengan format H x W x 3.")

    entropy_norm_values = []
    corr_abs_values = []

    for channel_idx in range(3):
        channel = C[:, :, channel_idx]

        # Entropi kanal, dinormalisasi terhadap nilai maksimum 8-bit
        entropy_ch = calculate_entropy(channel)
        entropy_norm_values.append(entropy_ch / 8.0)

        # Korelasi kanal
        corr_ch = calculate_correlation(channel)
        corr_abs_values.extend([
            abs(corr_ch["horizontal"]),
            abs(corr_ch["vertical"]),
            abs(corr_ch["diagonal"])
        ])

    H_tilde = float(np.mean(entropy_norm_values))
    rho_tilde = float(np.mean(corr_abs_values))

    return H_tilde, rho_tilde


def evaluate_fitness(P: np.ndarray, K_hex: str, X_candidate: np.ndarray) -> float:
    """
    Fungsi objektif untuk mengevaluasi satu kandidat parameter:
        X = [r_min, eps, T0, Q]

    Target: minimisasi.
    Semakin kecil nilai fitness, semakin baik kandidat parameter.

    Komponen:
    1. Penalti kelayakan parameter.
    2. Penalti kegagalan dekripsi lossless.
    3. Entropi RGB rata-rata.
    4. Korelasi absolut rata-rata RGB.
    """
    r_min, eps, T0, Q = X_candidate

    # 1. Evaluasi kendala kelayakan parameter
    r_max = 4.0 - eps

    if r_min >= r_max:
        return 1e10

    # 2. Setup parameter kandidat
    cfg = BaselineConfig(
        r_min=float(r_min),
        eps=float(eps),
        T0=int(round(T0)),
        Q=int(round(Q))
    )

    # 3. Uji coba enkripsi dan dekripsi
    try:
        C, _ = encrypt_baseline(P, K_hex, cfg=cfg, return_debug=False)
        P_hat = decrypt_baseline(C, K_hex, cfg=cfg)
    except Exception:
        return 1e10

    # 4. Penalti dekripsi lossless
    if not verify_lossless(P, P_hat):
        return 1e10

    # 5. Perhitungan komponen fitness berbasis RGB
    try:
        H_tilde, rho_tilde = _calculate_rgb_fitness_components(C)
    except Exception:
        return 1e10

    # 6. Pembobotan fungsi objektif
    w_H = 0.5
    w_rho = 0.5

    f_X = w_H * (1.0 - H_tilde) + w_rho * rho_tilde

    return float(f_X)