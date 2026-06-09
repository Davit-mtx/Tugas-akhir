
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
    eps: float = 0.02
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
    Membaca seluruh 128-bit kunci dan melipatnya (XOR) menjadi 8 subkey.
    Memastikan efek avalanche : perubahan 1 bit di ujung manapun akan merubah subkey
    """
    K_hex = K_hex.strip().lower().replace("0x", "")
    if len(K_hex) != 32:
        raise ValueError("K_hex harus 32 karakter hex (128-bit).")

    # Ambil seluruh 16 byte (32 karakter)
    bytes_array = [int(K_hex[i:i+2], 16) for i in range(0, 32, 2)]

    # lipat 16 byte menjadi 8 byte menggunakan operasi bitwise XOR
    subkeys = []
    for m in range(8):  # 16 pasangan hex? (32 hex char => 16 byte)
        folded_byte = bytes_array[m] ^ bytes_array[m+8]
        subkeys.append(folded_byte)

    # Paper/proposal kamu membagi jadi 8 sub-kunci K_m (masing-masing 2 hex).
    # Jika definisimu memang 8 subkey, biasanya diambil 8 byte pertama/tertentu.
    # Di banyak implementasi: K dibagi jadi 8 bagian, tiap bagian 2 hex (1 byte) => 8 byte.
    # Jadi kita ambil 8 byte pertama:
    Km_dec = np.array(subkeys, dtype=np.int64)
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

#@ tambahan pelengkap dan beberapa perbaikan pada bagian 
# =========================================================
# 4A) Helper khusus Oravec untuk tahap plaintext-related
# =========================================================

# Sesuai paper Oravec, transient pada tahap plaintext-related memakai 1000 iterasi.
ORAVEC_PR_TRANSIENT = 1000


def _oravec_quantize_max(seq: np.ndarray, max_value: int) -> np.ndarray:
    """
    Kuantisasi ala Oravec untuk menghasilkan bilangan bulat pada rentang 0..max_value.

    Ide penting:
    - empat digit desimal pertama "dibuang" dengan (10^4 * x) mod 1
    - lalu dipetakan ke domain diskret 0..max_value

    Hasil:
        q_n = floor( (max_value + 1) * ((10^4 * seq_n) mod 1) )
    """
    seq = np.asarray(seq, dtype=np.float64)
    frac = (1e4 * (seq % 1.0)) % 1.0
    q = np.floor((max_value + 1) * frac).astype(np.int64)
    return q


def _logistic_sequence_with_pattern(
    x0: float,
    r_pattern: np.ndarray,
    length: int,
    transient: int = 0,
) -> np.ndarray:
    """
    Bangkitkan deret logistic map ketika parameter r berubah mengikuti pola tertentu.

    Rumus penting:
        x_{n+1} = r_n * x_n * (1 - x_n)

    dengan r_n diambil secara siklik dari r_pattern.
    """
    r_pattern = np.asarray(r_pattern, dtype=np.float64)
    if r_pattern.ndim != 1 or r_pattern.size == 0:
        raise ValueError("r_pattern harus vektor 1D non-kosong.")

    total_steps = transient + length
    x = float(x0)
    out = np.empty(total_steps, dtype=np.float64)

    for t in range(total_steps):
        r_t = float(r_pattern[t % r_pattern.size])
        x = r_t * x * (1.0 - x)
        out[t] = x

    return out[transient:]


def _build_lt_row_major(rm: np.ndarray, H_: int, W_: int) -> np.ndarray:
    """
    Membuat lookup table LT ukuran H' x W' dengan pola berulang (r1..r8)
    menggunakan row-major order.
    """
    total = H_ * W_
    flat = np.resize(rm.astype(np.float64), total)
    LT = flat.reshape((H_, W_), order="C")
    return LT


def _shuffle_lt_oravec(LT: np.ndarray, rm: np.ndarray) -> np.ndarray:
    """
    Mengacak LT dengan dua tahap circular shift sesuai Oravec:
    1) shift kolom memakai seq1'
    2) shift baris memakai seq2'

    seq1:
      pattern = r4, r8, r3, r7, r2, r6, r1, r5
      length  = W'
      max     = H' - 1

    seq2:
      pattern = r5, r1, r6, r2, r7, r3, r8, r4
      length  = H'
      max     = W' - 1
    """
    H_, W_ = LT.shape

    #@ Pattern sesuai Table 1 paper Oravec
    seq1_pattern = np.array([rm[3], rm[7], rm[2], rm[6], rm[1], rm[5], rm[0], rm[4]], dtype=np.float64)
    seq2_pattern = np.array([rm[4], rm[0], rm[5], rm[1], rm[6], rm[2], rm[7], rm[3]], dtype=np.float64)

    #@ Generate deret chaos dengan x0 = 0.5 dan transient 1000
    seq1 = _logistic_sequence_with_pattern(
        x0=0.5,
        r_pattern=seq1_pattern,
        length=W_,
        transient=ORAVEC_PR_TRANSIENT,
    )
    seq2 = _logistic_sequence_with_pattern(
        x0=0.5,
        r_pattern=seq2_pattern,
        length=H_,
        transient=ORAVEC_PR_TRANSIENT,
    )

    #@ Kuantisasi ke rentang shift
    seq1_q = _oravec_quantize_max(seq1, max_value=H_ - 1)  # shift untuk kolom
    seq2_q = _oravec_quantize_max(seq2, max_value=W_ - 1)  # shift untuk baris

    #@ Oravec: kolom dulu, lalu baris
    LT_shuffled = _circular_shift_cols(LT, seq1_q)
    LT_shuffled = _circular_shift_rows(LT_shuffled, seq2_q)

    return LT_shuffled


def _get_oravec_x1001(rm: np.ndarray) -> float:
    """
    Menghasilkan x1001 untuk tahap plaintext-related.

    Sesuai paper:
    - x0 = 0.5
    - transient = 1000
    - pattern seq3 = r1, r2, ..., r8
    - hanya x1001 yang disimpan
    """
    seq3_pattern = np.array([rm[0], rm[1], rm[2], rm[3], rm[4], rm[5], rm[6], rm[7]], dtype=np.float64)

    x1001 = _logistic_sequence_with_pattern(
        x0=0.5,
        r_pattern=seq3_pattern,
        length=1,
        transient=ORAVEC_PR_TRANSIENT,
    )[0]

    return float(x1001)

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

    # Pecah kunci menjadi 4 block (masing-masing 8 karakter / 32-bit)
    block1 = int(K_hex_clean[0:8], 16)
    block2 = int(K_hex_clean[8:16], 16)
    block3 = int(K_hex_clean[16:24], 16)
    block4 = int(K_hex_clean[24:32], 16)

    # XOR-kan seluruh blok agar setiap bit berpartisipasi membentuk seed
    seed_int = block1 ^ block2 ^ block3 ^ block4  
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

def _add_mod256(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Penjumlahan modulo 256 pada domain uint8.
    Diproses di uint16 dulu agar tidak overflow sebelum modulo.
    """
    return ((a.astype(np.uint16) + b.astype(np.uint16)) % 256).astype(np.uint8)


def diffusion_oravec(P_in: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Four-dimensional diffusion stage ala Oravec.

    Catatan penting:
    - Core operator di sini mengikuti paper Oravec:
      1) top to bottom
      2) left to right
      3) bottom to top
      4) right to left
    - Pada tiap scan, vektor yang sedang diproses dikombinasikan dengan:
      * satu vektor via penjumlahan modulo 256
      * satu vektor via XOR
    - Saat ini S2 hanya divalidasi ukurannya dan dipertahankan di signature
      agar pipeline kamu tetap kompatibel dengan proposal.
      Karena proposal belum menuliskan operator S2 secara eksplisit, saya
      sengaja tidak "mengarang" pemakaian S2 di diffusion.
    """
    H_, W_ = P_in.shape
    N = H_ * W_

    if S2.size != N:
        raise ValueError(f"Ukuran S2 harus H'*W'={N}, dapat {S2.size}.")

    P = P_in.astype(np.uint8).copy()

    #Reshape S2 1D menjadi matriks 2D berukuran H' x W'
    S2_mat = S2.reshape((H_, W_)).astype(np.uint8)

    # -------------------------------------------------
    # Scan 1: top -> bottom
    # add : row l-1
    # xor : row l+1
    # wrap-around:
    #   l-1 < 0  -> H_-1
    #   l+1 >= H_ -> 0
    # -------------------------------------------------
    for l in range(H_):
        add_row = P[l - 1, :] if l > 0 else P[H_ - 1, :]
        xor_row = P[l + 1, :] if l < H_ - 1 else P[0, :]
        P[l, :] = _add_mod256(P[l, :], add_row) ^ xor_row ^ S2_mat[l,:]

    # -------------------------------------------------
    # Scan 2: left -> right
    # add : col k-1
    # xor : col k+1
    # wrap-around:
    #   k-1 < 0  -> W_-1
    #   k+1 >= W_ -> 0
    # -------------------------------------------------
    for k in range(W_):
        add_col = P[:, k - 1] if k > 0 else P[:, W_ - 1]
        xor_col = P[:, k + 1] if k < W_ - 1 else P[:, 0]
        P[:, k] = _add_mod256(P[:, k], add_col) ^ xor_col ^ S2_mat[:,k]

    # -------------------------------------------------
    # Scan 3: bottom -> top
    # add : row l+1
    # xor : row l-1
    # wrap-around sama
    # -------------------------------------------------
    for l in range(H_ - 1, -1, -1):
        add_row = P[l + 1, :] if l < H_ - 1 else P[0, :]
        xor_row = P[l - 1, :] if l > 0 else P[H_ - 1, :]
        P[l, :] = _add_mod256(P[l, :], add_row) ^ xor_row ^ S2_mat[l,:]

    # -------------------------------------------------
    # Scan 4: right -> left
    # add : col k+1
    # xor : col k-1
    # wrap-around sama
    # -------------------------------------------------
    for k in range(W_ - 1, -1, -1):
        add_col = P[:, k + 1] if k < W_ - 1 else P[:, 0]
        xor_col = P[:, k - 1] if k > 0 else P[:, W_ - 1]
        P[:, k] = _add_mod256(P[:, k], add_col) ^ xor_col ^ S2_mat[:,k]

    return P

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


def plaintext_related_encrypt(
    Pprime: np.ndarray,
    rm: np.ndarray,
    cfg: BaselineConfig,
) -> np.ndarray:
    """
    Plaintext-related row-wise encryption sesuai Oravec.

    Langkah besar:
    1) Bangun LT ukuran H' x W' dengan pola (r1..r8) row-major
    2) Shuffle LT:
       - column circular shift dengan seq1'
       - row circular shift dengan seq2'
    3) Bangkitkan x1001 dari seq3 (x0=0.5, transient=1000)
    4) Untuk tiap baris l:
       - modifikasi LT(l,:) dengan plaintext row sebelumnya
       - bangkitkan seqplr sepanjang W' TANPA transient tambahan
       - kuantisasi ke 0..255
       - XOR dengan baris plaintext saat ini
    """
    H_, W_ = Pprime.shape
    Pprime = Pprime.astype(np.uint8)

    # -------------------------------------------------
    # 1) Build LT dengan row-major repeating (r1..r8)
    # -------------------------------------------------
    LT = _build_lt_row_major(rm, H_, W_)

    # -------------------------------------------------
    # 2) Shuffle LT sesuai Oravec
    # -------------------------------------------------
    LT = _shuffle_lt_oravec(LT, rm)

    # -------------------------------------------------
    # 3) Ambil x1001 dari seq3
    # -------------------------------------------------
    x1001 = _get_oravec_x1001(rm)

    # -------------------------------------------------
    # 4) Row-wise plaintext-related encryption
    # -------------------------------------------------
    out = np.empty_like(Pprime, dtype=np.uint8)

    for l in range(H_):
        # Untuk baris pertama, gunakan wrap-around: baris sebelumnya = baris terakhir
        prev_row = Pprime[l - 1, :] if l > 0 else np.full(W_, 128, dtype=np.uint8)

        # BARIS PENTING:
        # LT(l,:) = LT(l,:) + 10^-15 * 65536 * P'(l-1,:)
        LT_row_mod = LT[l, :] + (10.0 ** -15) * 65536.0 * prev_row.astype(np.float64)

        # Bangkitkan seqplr sepanjang W' tanpa transient tambahan,
        # dengan initial value x1001 dan parameter berubah mengikuti LT_row_mod
        seqplr = _logistic_sequence_with_pattern(
            x0=x1001,
            r_pattern=LT_row_mod,
            length=W_,
            transient=0,
        )

        # Kuantisasi ke byte 0..255
        ks_row = _oravec_quantize_max(seqplr, max_value=255).astype(np.uint8)

        # BARIS PENTING:
        # P^(pr)(l,:) = P'(l,:) XOR seq'_plr
        out[l, :] = Pprime[l, :] ^ ks_row

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
    # 3) Plaintext-related 
    #    TODO: ganti dengan implementasi LT + update parameter (2.4) + XOR (2.5)
    # ---------------------------
    Ppr = plaintext_related_encrypt(Pprime, rm, cfg)
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
    # 6) Diffusion oravec
    #    TODO: ganti dengan diffusion multi-arah sesuai proposal
    # ---------------------------
    Pdiff = diffusion_oravec(Pconf, S2)
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


# =========================================================
# =========================================================
#                   DECRYPTION PIPELINE
# =========================================================
# =========================================================

def _sub_mod256(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pengurangan modulo 256 pada domain uint8.
    Invers deterministik dari _add_mod256.
    """
    return ((a.astype(np.int16) - b.astype(np.int16)) % 256).astype(np.uint8)

# ---------------------------------------------------------
# 1. Inverse Diffusion
# ---------------------------------------------------------
def inverse_diffusion_oravec(P_in: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Invers dari 4D diffusion Oravec.
    Pemindaian dilakukan dengan urutan terbalik (Scan 4 -> 3 -> 2 -> 1)
    dan arah iterasinya juga dibalik sempurna.
    """
    H_, W_ = P_in.shape
    P = P_in.astype(np.uint8).copy()
    S2_mat = S2.reshape((H_, W_)).astype(np.uint8)

    # Reversing Scan 4 (Kanan -> Kiri). Dekripsi berbalik jadi (Kiri -> Kanan)
    for k in range(W_):
        add_col = P[:, k + 1] if k < W_ - 1 else P[:, 0]
        xor_col = P[:, k - 1] if k > 0 else P[:, W_ - 1]
        P[:, k] = _sub_mod256(P[:, k] ^ xor_col ^ S2_mat[:, k], add_col)

    # Reversing Scan 3 (Bawah -> Atas). Dekripsi berbalik jadi (Atas -> Bawah)
    for l in range(H_):
        add_row = P[l + 1, :] if l < H_ - 1 else P[0, :]
        xor_row = P[l - 1, :] if l > 0 else P[H_ - 1, :]
        P[l, :] = _sub_mod256(P[l, :] ^ xor_row ^ S2_mat[l, :], add_row)

    # Reversing Scan 2 (Kiri -> Kanan). Dekripsi berbalik jadi (Kanan -> Kiri)
    for k in range(W_ - 1, -1, -1):
        add_col = P[:, k - 1] if k > 0 else P[:, W_ - 1]
        xor_col = P[:, k + 1] if k < W_ - 1 else P[:, 0]
        P[:, k] = _sub_mod256(P[:, k] ^ xor_col ^ S2_mat[:, k], add_col)

    # Reversing Scan 1 (Atas -> Bawah). Dekripsi berbalik jadi (Bawah -> Atas)
    for l in range(H_ - 1, -1, -1):
        add_row = P[l - 1, :] if l > 0 else P[H_ - 1, :]
        xor_row = P[l + 1, :] if l < H_ - 1 else P[0, :]
        P[l, :] = _sub_mod256(P[l, :] ^ xor_row ^ S2_mat[l, :], add_row)

    return P

# ---------------------------------------------------------
# 2. Inverse Confusion
# ---------------------------------------------------------
def inverse_confusion_circular_shift(P_in: np.ndarray, S1: np.ndarray) -> np.ndarray:
    """
    Invers dari confusion. Urutan shift dibalik: baris dulu, baru kolom.
    Arah shift di-negasikan (minus).
    """
    H_, W_ = P_in.shape
    S1_row = (S1[:H_].astype(np.int64) % W_)
    S1_col = (S1[H_:].astype(np.int64) % H_)

    # Undo row shifts then col shifts
    out = _circular_shift_rows(P_in, -S1_row)
    out = _circular_shift_cols(out, -S1_col)
    return out

# ---------------------------------------------------------
# 3. Inverse Plaintext-Related
# ---------------------------------------------------------
def inverse_plaintext_related(Ppr: np.ndarray, rm: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    H_, W_ = Ppr.shape
    Pprime_dec = np.empty_like(Ppr, dtype=np.uint8)

    LT = _build_lt_row_major(rm, H_, W_)
    LT = _shuffle_lt_oravec(LT, rm)
    x1001 = _get_oravec_x1001(rm)

    for l in range(H_):
        # KUNCI DEKRIPSI: Baris 0 menggunakan IV statis (angka 128) agar tidak terjadi deadlock
        prev_row = Pprime_dec[l - 1, :] if l > 0 else np.full(W_, 128, dtype=np.uint8)

        LT_row_mod = LT[l, :] + (10.0 ** -15) * 65536.0 * prev_row.astype(np.float64)
        
        seqplr = _logistic_sequence_with_pattern(x0=x1001, r_pattern=LT_row_mod, length=W_, transient=0)
        ks_row = _oravec_quantize_max(seqplr, max_value=255).astype(np.uint8)

        # XOR lagi untuk mendapatkan Pprime_dec (karena XOR bersifat involutif)
        Pprime_dec[l, :] = Ppr[l, :] ^ ks_row

    return Pprime_dec

# ---------------------------------------------------------
# 4. Fungsi Utama: Decrypt Baseline
# ---------------------------------------------------------
def decrypt_baseline(C: np.ndarray, K_hex: str, cfg: Optional[BaselineConfig] = None) -> np.ndarray:
    if cfg is None:
        cfg = BaselineConfig()

    Km_dec = split_key_128hex_to_subkeys(K_hex)
    rm = rm_from_subkeys_oravec(Km_dec)

    # 1. Forward rearrange cipher image to get correct matrix format and meta
    # Ini trik penting karena fungsi rearrange_image mendatar-kan RGB menjadi matriks H' x W'
    Pout, meta = rearrange_image(C)
    H_, W_ = Pout.shape

    S1, S2, S3 = build_S1_S2_S3(K_hex, cfg, H_, W_)

    # Step 1: Inverse Whitening (XOR)
    Pdiff = key_whitening(Pout, S3)

    # Step 2: Inverse Diffusion
    Pconf = inverse_diffusion_oravec(Pdiff, S2)

    # Step 3: Inverse Confusion
    Ppr = inverse_confusion_circular_shift(Pconf, S1)

    # Step 4: Inverse Plaintext-Related
    Pprime_dec = inverse_plaintext_related(Ppr, rm, cfg)

    # Step 5: Rekonstruksi ke Citra Asli
    P_dec = inverse_rearrange_to_image(Pprime_dec, meta)

    return P_dec