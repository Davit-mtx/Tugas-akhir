
"""
Baseline Encryption Pipeline (Oravec-style) — ENCRYPTION ONLY
===========================================================

Tujuan file ini:
- Memberikan kerangka kode Python yang rapi untuk pipeline enkripsi baseline.
- Setiap tahap besar diberi komentar jelas.
- Baris-baris penting yang merepresentasikan rumus diberi komentar khusus.

Catatan penting:
- Beberapa detail skema (khususnya DIFFUSION dan pembentukan S1/S2/S3 secara persis)
  bisa berbeda antar implementasi/paper. Di sini saya buat kerangka + default yang aman,
  dan saya tandai bagian yang HARUS kamu sesuaikan dengan proposalmu.

Kamu bisa taruh ini di: src/crypto/pipeline.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


# =========================================================
# 0) Konfigurasi baseline (X0)
# =========================================================

@dataclass(frozen=True)
class BaselineConfig:
    r_min: float = 3.70
    eps: float = 0.002
    T0: int = 500
    Q: int = 256

    @property
    def r_max(self) -> float:
        return 4.0 - self.eps


# =========================================================
# 1) Util dasar: validasi & helper
# =========================================================

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _circular_shift_1d(arr: np.ndarray, shift: int) -> np.ndarray:
    """Circular shift 1D: shift > 0 geser ke kanan (default numpy roll)."""
    shift = int(shift)
    if arr.size == 0:
        return arr
    return np.roll(arr, shift % arr.size)


def _circular_shift_rows(mat: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Circular shift tiap baris i dengan shifts[i]."""
    out = mat.copy()
    for i in range(mat.shape[0]):
        out[i, :] = _circular_shift_1d(out[i, :], int(shifts[i]))
    return out


def _circular_shift_cols(mat: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Circular shift tiap kolom j dengan shifts[j]."""
    out = mat.copy()
    for j in range(mat.shape[1]):
        out[:, j] = _circular_shift_1d(out[:, j], int(shifts[j]))
    return out


# =========================================================
# 2) Key Processing: K (128-bit hex) -> subkey Km -> r_m
# =========================================================

def split_key_128hex_to_subkeys(K_hex: str) -> np.ndarray:
    """
    Input: K_hex string heksadesimal 128-bit (32 hex chars).
    Output: Km_dec array shape (8,), masing-masing integer 0..255 (2 hex digits per subkey).

    Rumus penting:
    Km = 16 * Km(1) + Km(2)
    di mana Km(1), Km(2) adalah digit hex (0..15).
    """
    K_hex = K_hex.strip().lower().replace("0x", "")
    if len(K_hex) != 32:
        raise ValueError("K_hex harus 32 karakter hex (128-bit).")

    subkeys = []
    for m in range(0, 16, 2):  # 16 pasangan hex? (32 hex char => 16 byte)
        byte_hex = K_hex[m:m+2]
        subkeys.append(int(byte_hex, 16))

    # Paper/proposal kamu membagi jadi 8 sub-kunci K_m (masing-masing 2 hex).
    # Jika definisimu memang 8 subkey, biasanya diambil 8 byte pertama/tertentu.
    # Di banyak implementasi: K dibagi jadi 8 bagian, tiap bagian 2 hex (1 byte) => 8 byte.
    # Jadi kita ambil 8 byte pertama:
    Km_dec = np.array(subkeys[:8], dtype=np.int64)
    return Km_dec


def rm_from_subkeys_oravec(Km_dec: np.ndarray) -> np.ndarray:
    """
    Membentuk r_m (m=1..8) dari K_m sesuai gaya Oravec.
    Rumus penting (sesuai proposal kamu):

    r_m = 4 - 10^{-15} * ( (9-m)*256*65536 - K_m ), m=1..8

    Catatan:
    - Ini menghasilkan r_m sangat dekat 4.
    - Pastikan dtype float64 untuk presisi.
    """
    if Km_dec.shape != (8,):
        raise ValueError("Km_dec harus shape (8,).")

    rm = np.zeros(8, dtype=np.float64)
    for idx in range(8):
        m = idx + 1
        # ====== BARIS PENTING: merepresentasikan persamaan r_m ======
        rm[idx] = 4.0 - (10.0 ** -15) * (((9 - m) * 256 * 65536) - float(Km_dec[idx]))
    return rm


# =========================================================
# 3) Image Rearrangement: RGB -> P' (H' x W')
# =========================================================

@dataclass
class RearrangementMeta:
    mode: str            # "gray" atau "rgb_interleave_cols"
    H: int
    W: int
    C: int               # channels: 1 atau 3


def rearrange_image(P: np.ndarray) -> Tuple[np.ndarray, RearrangementMeta]:
    """
    Output internal matrix P' berukuran H' x W' (uint8).

    - Jika grayscale (H,W) -> P' = P
    - Jika RGB (H,W,3) -> kolom interleave: [R_col0, G_col0, B_col0, R_col1, G_col1, B_col1, ...]
      sehingga W' = 3*W dan H' = H.

    Ini harus BIJEKTIF supaya bisa dibalik saat dekripsi.
    """
    P = _ensure_uint8(P)

    if P.ndim == 2:
        H, W = P.shape
        meta = RearrangementMeta(mode="gray", H=H, W=W, C=1)
        return P.copy(), meta

    if P.ndim == 3 and P.shape[2] == 3:
        H, W, C = P.shape
        R = P[:, :, 0]
        G = P[:, :, 1]
        B = P[:, :, 2]

        # Interleave columns: untuk tiap col j, hasilkan 3 kolom [R[:,j], G[:,j], B[:,j]]
        cols = []
        for j in range(W):
            cols.append(R[:, j:j+1])
            cols.append(G[:, j:j+1])
            cols.append(B[:, j:j+1])
        Pprime = np.concatenate(cols, axis=1)  # (H, 3W)

        meta = RearrangementMeta(mode="rgb_interleave_cols", H=H, W=W, C=3)
        return Pprime.astype(np.uint8), meta

    raise ValueError("Format image tidak didukung. Gunakan (H,W) atau (H,W,3).")


# =========================================================
# 4) Chaos generator + kuantisasi (untuk S1/S2/S3)
# =========================================================

def logistic_map_sequence(x0: float, r: float, n: int) -> np.ndarray:
    """
    Bangkitkan deret logistic map:
        x_{t+1} = r * x_t * (1 - x_t)
    Output float64 (0..1).
    """
    x = np.empty(n, dtype=np.float64)
    x[0] = x0
    for t in range(n - 1):
        x[t + 1] = r * x[t] * (1.0 - x[t])
    return x


def quantize_sequence(seq: np.ndarray, Q: int) -> np.ndarray:
    """
    Kuantisasi ke bilangan diskret.

    Rumus penting (sesuai proposal kamu, konsep umum):
        seq'_n = floor( (1/Q) * 10^4 * (seq_n mod 1) )

    Catatan:
    - seq logistic map sudah di (0,1), jadi (mod 1) tidak mengubah.
    - Hasil masih perlu dipetakan ke 0..255 jika kamu butuh byte.
    """
    seq = np.asarray(seq, dtype=np.float64)
    # ====== BARIS PENTING: merepresentasikan persamaan kuantisasi ======
    q = np.floor((1.0 / float(Q)) * 1e4 * (seq % 1.0)).astype(np.int64)
    return q


def build_S1_S2_S3(K_hex: str, cfg: BaselineConfig, H_: int, W_: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Membentuk S1, S2, S3 untuk confusion/diffusion/whitening.

    Karena detail pemotongan/concatenation S1/S2/S3 bisa berbeda di proposal,
    saya pakai pendekatan deterministik berikut:
    - Ambil seed x0 dari K_hex (hash sederhana) => x0 di (0,1)
    - Gunakan r = cfg.r_max (atau rata-rata r_m) untuk deret global
    - Bangkitkan panjang L = H_*W_ + (H_ + W_) + H_*W_ (cukup untuk S1,S2,S3)
    - Kuantisasi, lalu map ke byte 0..255 via modulo 256.

    Kamu boleh ganti pendekatan ini agar 1:1 dengan definisi proposalmu.
    """
    # Seed deterministik dari key
    K_hex_clean = K_hex.strip().lower().replace("0x", "")
    seed_int = int(K_hex_clean[:8], 16)  # ambil 32-bit awal
    x0 = ((seed_int % 10_000_000) + 1) / 10_000_001.0  # (0,1)

    # Panjang total yang dibutuhkan (aman)
    L_s1 = H_ + W_              # shifts untuk baris + kolom
    L_s2 = H_ * W_              # diffusion control
    L_s3 = H_ * W_              # whitening mask
    L_total = cfg.T0 + (L_s1 + L_s2 + L_s3) + 5

    seq = logistic_map_sequence(x0=x0, r=cfg.r_max, n=L_total)
    seq_use = seq[cfg.T0:]  # buang transient global

    q = quantize_sequence(seq_use, cfg.Q)
    q_byte = (q % 256).astype(np.uint8)

    # Split deterministik
    s1 = q_byte[:L_s1]
    s2 = q_byte[L_s1:L_s1 + L_s2]
    s3 = q_byte[L_s1 + L_s2:L_s1 + L_s2 + L_s3]

    return s1, s2, s3


# =========================================================
# 5) Confusion: circular shifts dengan S1
# =========================================================

def confusion_circular_shift(P_in: np.ndarray, S1: np.ndarray) -> np.ndarray:
    """
    Confusion Oravec-style: circular shift kolom dan baris.

    S1 dibagi:
      - S1_row: panjang H' untuk shift tiap baris
      - S1_col: panjang W' untuk shift tiap kolom

    Catatan:
    - Arah shift harus konsisten dengan definisi dekripsi nanti.
    - Di sini: shift positif => np.roll (ke kanan untuk baris, ke bawah untuk kolom).
      Kamu boleh ubah arah jika proposalmu mendefinisikan sebaliknya.
    """
    H_, W_ = P_in.shape
    if S1.size != H_ + W_:
        raise ValueError(f"Ukuran S1 harus H'+W'={H_+W_}, dapat {S1.size}.")

    S1_row = (S1[:H_].astype(np.int64) % W_)  # shift baris mod W'
    S1_col = (S1[H_:].astype(np.int64) % H_)  # shift kolom mod H'

    # ===== Tahap besar: shift kolom lalu shift baris (atau sebaliknya sesuai proposal) =====
    out = _circular_shift_cols(P_in, S1_col)
    out = _circular_shift_rows(out, S1_row)
    return out


# =========================================================
# 6) Diffusion: placeholder (HARUS kamu samakan dengan proposalmu)
# =========================================================

def diffusion_placeholder(P_in: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    WARNING:
    Diffusion adalah bagian yang paling sering berbeda detailnya.
    Di proposal kamu ada "multi-arah + XOR/mod 256" dan dikendalikan S2.
    Karena rumus persisnya belum kamu paste di sini, saya buat diffusion minimal deterministik:

    - Flatten P menjadi 1D row-major.
    - Scan forward: y[i] = (x[i] XOR S2[i]) XOR y[i-1]
    - Lalu scan backward: y[i] = (y[i] XOR S2[i]) XOR y[i+1]
    - Reshape kembali.

    Ini menghasilkan efek difusi (perubahan satu piksel mempengaruhi banyak output),
    dan masih invertible (dengan definisi inverse yang tepat).

    Nanti kamu tinggal GANTI fungsi ini dengan diffusion yang benar sesuai proposal.
    """
    H_, W_ = P_in.shape
    N = H_ * W_
    if S2.size != N:
        raise ValueError(f"Ukuran S2 harus H'*W'={N}, dapat {S2.size}.")

    x = P_in.flatten(order="C").astype(np.uint8)
    k = S2.astype(np.uint8)

    y = np.empty_like(x)

    # ===== Scan maju (forward diffusion) =====
    prev = np.uint8(0)
    for i in range(N):
        # BARIS PENTING: XOR chaining (difusi)
        y[i] = (x[i] ^ k[i]) ^ prev
        prev = y[i]

    # ===== Scan mundur (backward diffusion) =====
    z = np.empty_like(y)
    nxt = np.uint8(0)
    for i in range(N - 1, -1, -1):
        # BARIS PENTING: XOR chaining reverse
        z[i] = (y[i] ^ k[i]) ^ nxt
        nxt = z[i]

    return z.reshape((H_, W_), order="C").astype(np.uint8)


# =========================================================
# 7) Key Whitening: XOR dengan mask dari S3
# =========================================================

def key_whitening(P_in: np.ndarray, S3: np.ndarray) -> np.ndarray:
    """
    Whitening:
      P_out = P_in XOR reshape(S3, H', W')
    """
    H_, W_ = P_in.shape
    if S3.size != H_ * W_:
        raise ValueError(f"Ukuran S3 harus H'*W'={H_*W_}, dapat {S3.size}.")

    mask = S3.reshape((H_, W_), order="C").astype(np.uint8)

    # ===== BARIS PENTING: XOR whitening sesuai persamaan proposal =====
    return (P_in.astype(np.uint8) ^ mask).astype(np.uint8)


# =========================================================
# 8) Plaintext-related row-wise (placeholder terstruktur)
# =========================================================

def plaintext_related_encrypt_placeholder(
    Pprime: np.ndarray,
    rm: np.ndarray,
    cfg: BaselineConfig,
    K_hex: str,
) -> np.ndarray:
    """
    WARNING:
    Tahap plaintext-related Oravec punya lookup table LT dan pembangkitan keystream per baris,
    dengan modifikasi parameter per baris:
        LT(a,:) = LT(a,:) + 10^{-15}*65536*P'(a-1,:)

    Karena definisi detail LT shuffle + cara membentuk KS(a,:) dari LT(a,:) cukup spesifik,
    saya buat placeholder yang tetap:
    - deterministik
    - menghasilkan output berbeda dari input
    - mudah diganti dengan implementasi final sesuai proposal

    Placeholder ini:
    - membangkitkan keystream per baris dari logistic map global berbasis key
    - XOR-kan ke setiap baris

    Nanti kamu GANTI isi fungsi ini dengan implementasi LT + update parameter (2.4) + XOR (2.5).
    """
    H_, W_ = Pprime.shape

    # Buat seed x0 deterministik dari key
    K_hex_clean = K_hex.strip().lower().replace("0x", "")
    seed_int = int(K_hex_clean[-8:], 16)  # ambil 32-bit akhir
    x0 = ((seed_int % 10_000_000) + 1) / 10_000_001.0

    # Pakai r rata-rata sebagai placeholder (nanti ganti pakai LT(a,:))
    r = float(np.mean(rm))

    out = Pprime.copy().astype(np.uint8)

    for a in range(H_):
        # Bangkitkan deret sepanjang W_ + T0
        seq = logistic_map_sequence(x0=x0, r=r, n=cfg.T0 + W_ + 1)[cfg.T0:cfg.T0 + W_]
        ks_int = (quantize_sequence(seq, cfg.Q) % 256).astype(np.uint8)

        # ===== BARIS PENTING: XOR row-wise (sesuai struktur persamaan 2.5) =====
        out[a, :] = out[a, :] ^ ks_int

        # update x0 agar baris berbeda (placeholder)
        x0 = float(seq[-1])

    return out


# =========================================================
# 9) Inverse rearrangement untuk menghasilkan cipher-image format
# =========================================================

def inverse_rearrange_to_image(Pout: np.ndarray, meta: RearrangementMeta) -> np.ndarray:
    """
    Mengubah internal matrix kembali ke format citra output:
    - gray: (H,W)
    - rgb_interleave_cols: (H,W,3)
    """
    if meta.mode == "gray":
        return Pout.astype(np.uint8)

    if meta.mode == "rgb_interleave_cols":
        H, W = meta.H, meta.W
        # Pout shape (H, 3W)
        if Pout.shape != (H, 3 * W):
            raise ValueError("Ukuran Pout tidak sesuai meta RGB interleave.")

        R = np.empty((H, W), dtype=np.uint8)
        G = np.empty((H, W), dtype=np.uint8)
        B = np.empty((H, W), dtype=np.uint8)

        # Ambil kolom 0,1,2 untuk col0; 3,4,5 untuk col1; dst.
        for j in range(W):
            R[:, j] = Pout[:, 3 * j + 0]
            G[:, j] = Pout[:, 3 * j + 1]
            B[:, j] = Pout[:, 3 * j + 2]

        Cimg = np.stack([R, G, B], axis=2)
        return Cimg.astype(np.uint8)

    raise ValueError("Mode rearrangement tidak dikenal.")


# =========================================================
# 10) ENCRYPT BASELINE: gabungkan semua tahap
# =========================================================

def encrypt_baseline(
    P: np.ndarray,
    K_hex: str,
    cfg: Optional[BaselineConfig] = None,
    return_debug: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Pipeline enkripsi baseline (encryption only).
    Output:
      - C: cipher-image (format sama dengan input image: gray atau RGB)
      - debug: dict intermediate (opsional, untuk tracking di notebook)

    Urutan tahap besar:
      1) Key processing -> rm
      2) Rearrangement -> P'
      3) Plaintext-related row-wise -> P_pr
      4) Global chaos -> S1,S2,S3
      5) Confusion -> P_conf
      6) Diffusion -> P_diff
      7) Whitening -> P_out
      8) Inverse rearrange -> C
    """
    if cfg is None:
        cfg = BaselineConfig()

    debug: Dict[str, np.ndarray] = {}

    # ---------------------------
    # 1) Key processing
    # ---------------------------
    Km_dec = split_key_128hex_to_subkeys(K_hex)
    rm = rm_from_subkeys_oravec(Km_dec)
    if return_debug:
        debug["Km_dec"] = Km_dec.astype(np.int64)
        debug["rm"] = rm.copy()

    # ---------------------------
    # 2) Rearrangement
    # ---------------------------
    Pprime, meta = rearrange_image(P)
    if return_debug:
        debug["Pprime"] = Pprime.copy()

    H_, W_ = Pprime.shape

    # ---------------------------
    # 3) Plaintext-related (placeholder)
    #    TODO: ganti dengan implementasi LT + update parameter (2.4) + XOR (2.5)
    # ---------------------------
    Ppr = plaintext_related_encrypt_placeholder(Pprime, rm, cfg, K_hex)
    if return_debug:
        debug["Ppr"] = Ppr.copy()

    # ---------------------------
    # 4) Global chaos -> S1,S2,S3
    #    TODO: samakan cara generate global sequence + split dengan proposal kamu
    # ---------------------------
    S1, S2, S3 = build_S1_S2_S3(K_hex, cfg, H_, W_)
    if return_debug:
        debug["S1"] = S1.copy()
        debug["S2"] = S2.copy()
        debug["S3"] = S3.copy()

    # ---------------------------
    # 5) Confusion (circular shifts)
    # ---------------------------
    Pconf = confusion_circular_shift(Ppr, S1)
    if return_debug:
        debug["Pconf"] = Pconf.copy()

    # ---------------------------
    # 6) Diffusion (placeholder)
    #    TODO: ganti dengan diffusion multi-arah sesuai proposal
    # ---------------------------
    Pdiff = diffusion_placeholder(Pconf, S2)
    if return_debug:
        debug["Pdiff"] = Pdiff.copy()

    # ---------------------------
    # 7) Key whitening
    # ---------------------------
    Pout = key_whitening(Pdiff, S3)
    if return_debug:
        debug["Pout"] = Pout.copy()

    # ---------------------------
    # 8) Inverse rearrangement -> cipher image
    # ---------------------------
    C = inverse_rearrange_to_image(Pout, meta)
    if return_debug:
        debug["C"] = C.copy()

    return C, debug


# =========================================================
# Contoh pemakaian cepat (untuk notebook / smoke test)
# =========================================================
if __name__ == "__main__":
    # Dummy test pakai matriks random 16x16 grayscale
    rng = np.random.default_rng(123)
    P = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)

    # Contoh key 128-bit hex (32 chars)
    K = "00112233445566778899aabbccddeeff"

    C, dbg = encrypt_baseline(P, K, return_debug=True)
    print("Plain shape:", P.shape, "Cipher shape:", C.shape)
    print("Stages:", list(dbg.keys()))
