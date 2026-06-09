
import numpy as np

def calculate_entropy(image_channel: np.ndarray) -> float:
    """
    Menghitung Entropi Shannon (Persamaan 2.34).
    Target ideal untuk citra 8-bit adalah sangat dekat dengan 8.0.
    """
    # Hitung histogram/frekuensi kemunculan tiap nilai piksel (0-255)
    hist, _ = np.histogram(image_channel.flatten(), bins=256, range=(0, 256))
    
    # Hitung probabilitas (hindari probabilitas 0 agar log2 tidak error)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    
    # Rumus Entropi Shannon
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)

def calculate_correlation(image_channel: np.ndarray) -> dict:
    """
    Menghitung Korelasi Piksel Bertetangga (Horizontal, Vertikal, Diagonal) (Persamaan 2.35).
    Target ideal untuk citra terenkripsi adalah mendekati 0.
    """
    img = image_channel.astype(np.float64)
    
    # Memisahkan pasangan piksel yang bertetangga
    # Horizontal (kiri & kanan)
    x_h = img[:, :-1].flatten()
    y_h = img[:, 1:].flatten()
    
    # Vertikal (atas & bawah)
    x_v = img[:-1, :].flatten()
    y_v = img[1:, :].flatten()
    
    # Diagonal (kiri atas & kanan bawah)
    x_d = img[:-1, :-1].flatten()
    y_d = img[1:, 1:].flatten()
    
    # Menghitung korelasi Pearson
    corr_h = np.corrcoef(x_h, y_h)[0, 1]
    corr_v = np.corrcoef(x_v, y_v)[0, 1]
    corr_d = np.corrcoef(x_d, y_d)[0, 1]
    
    return {"horizontal": corr_h, "vertical": corr_v, "diagonal": corr_d}

def calculate_npcr_uaci(C1_channel: np.ndarray, C2_channel: np.ndarray) -> tuple:
    """
    Menghitung NPCR (Persamaan 2.38) dan UACI (Persamaan 2.39).
    Digunakan untuk menguji ketahanan terhadap Differential Attack.
    C1 dan C2 adalah dua cipher-image yang dihasilkan dari plaintext yang berbeda 1 piksel.
    Target ideal: NPCR > 99.6%, UACI sekitar 33.4%.
    """
    c1 = C1_channel.astype(np.float64)
    c2 = C2_channel.astype(np.float64)
    
    # Hitung jumlah piksel yang berbeda
    diff_indicator = (c1 != c2).astype(np.float64)
    npcr = np.mean(diff_indicator) * 100.0
    
    # Hitung rata-rata selisih intensitas piksel
    uaci = np.mean(np.abs(c1 - c2) / 255.0) * 100.0
    
    return float(npcr), float(uaci)

def verify_lossless(P: np.ndarray, P_hat: np.ndarray) -> bool:
    """
    Memverifikasi keberhasilan dekripsi mutlak P == P^ (Persamaan 2.42).
    """
    return np.array_equal(P, P_hat)