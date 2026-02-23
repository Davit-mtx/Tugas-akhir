# 📚 DOKUMENTASI PIPELINE ENKRIPSI BASELINE (Oravec-style)

## **OVERVIEW**

File `pipeline.py` mengimplementasikan **ALGORITMA ENKRIPSI CITRA** berbasis chaos dengan metode Oravec. Pipeline ini dirancang khusus untuk **ENKRIPSI SAJA** (tidak termasuk dekripsi).

### **Tujuan Utama:**
- ✅ Mengenkripsi citra digital (grayscale atau RGB)
- ✅ Menggunakan kunci 128-bit dalam format heksadesimal
- ✅ Menerapkan teknik chaos-based encryption dengan logistic map
- ✅ Memberikan kerangka yang modular dan mudah dikustomisasi

---

## **🔑 KONSEP KUNCI**

### **Parameter Konfigurasi Baseline**
Class: `BaselineConfig`

| Parameter | Nilai Default | Deskripsi |
|-----------|---------------|-----------|
| `r_min` | 3.70 | Nilai minimum parameter chaos |
| `r_max` | 3.998 | Nilai maksimum (4.0 - eps) |
| `eps` | 0.002 | Epsilon untuk stabilitas |
| `T0` | 500 | **Jumlah iterasi transient** yang dibuang |
| `Q` | 256 | **Parameter kuantisasi** untuk diskretisasi |

> **PENTING:** `r_max` dihitung otomatis sebagai `4.0 - eps` untuk menjaga kestabilan logistic map di dekat kawasan chaos.

---

## **🏗️ ARSITEKTUR PIPELINE ENKRIPSI**

Pipeline enkripsi terdiri dari **8 TAHAP UTAMA** yang dieksekusi secara berurutan:

```
INPUT: Citra P (grayscale/RGB) + Kunci K (128-bit hex)
   ↓
[1] KEY PROCESSING: K → Km → rm
   ↓
[2] IMAGE REARRANGEMENT: P → P' (H' × W')
   ↓
[3] PLAINTEXT-RELATED ROW-WISE: P' → Ppr
   ↓
[4] GLOBAL CHAOS GENERATION: K → S1, S2, S3
   ↓
[5] CONFUSION (Circular Shifts): Ppr → Pconf
   ↓
[6] DIFFUSION (Multi-directional): Pconf → Pdiff
   ↓
[7] KEY WHITENING (XOR Masking): Pdiff → Pout
   ↓
[8] INVERSE REARRANGEMENT: Pout → C
   ↓
OUTPUT: Cipher-image C
```

---

## **📖 PENJELASAN DETAIL SETIAP TAHAP**

### **[TAHAP 1] KEY PROCESSING**

#### **Fungsi:** `split_key_128hex_to_subkeys(K_hex)`

**Input:** String heksadesimal 128-bit (32 karakter)  
**Output:** Array Km dengan 8 subkey (masing-masing 0-255)

**RUMUS PENTING:**
```
Km = 16 × Km(1) + Km(2)
```
- **Km(1):** Digit hex pertama (high nibble)
- **Km(2):** Digit hex kedua (low nibble)

**Contoh:**
```
K = "00112233445566778899aabbccddeeff"
→ K[0:2] = "00" → Km[0] = 0
→ K[2:4] = "11" → Km[1] = 17
... (8 subkey total)
```

---

#### **Fungsi:** `rm_from_subkeys_oravec(Km_dec)`

**Input:** Array Km (8 subkeys)  
**Output:** Array rm (8 parameter chaos)

**RUMUS KUNCI (Oravec):**
```
r_m = 4 - 10^(-15) × ((9-m) × 256 × 65536 - K_m)
```
dimana **m = 1, 2, ..., 8**

> **CATATAN PENTING:** Rumus ini menghasilkan nilai `r_m` yang **SANGAT DEKAT DENGAN 4** (nilai chaos kuat pada logistic map). Perbedaan kecil dari kunci menciptakan keyspace yang besar.

---

### **[TAHAP 2] IMAGE REARRANGEMENT**

#### **Fungsi:** `rearrange_image(P)`

**Tujuan:** Mengubah format citra menjadi matriks internal **P'** yang siap diproses.

**MODE OPERASI:**

1. **Grayscale (H × W):**
   ```
   P' = P (tidak ada perubahan)
   H' = H, W' = W
   ```

2. **RGB (H × W × 3):**
   ```
   COLUMN INTERLEAVING:
   [R_col0 | G_col0 | B_col0 | R_col1 | G_col1 | B_col1 | ...]
   
   H' = H
   W' = 3 × W
   ```

> **⚠️ CRITICAL:** Rearrangement harus **BIJEKTIF** (reversible) agar dekripsi bisa mengembalikan format original.

---

### **[TAHAP 3] PLAINTEXT-RELATED ROW-WISE**

#### **Fungsi:** `plaintext_related_encrypt_placeholder(Pprime, rm, cfg, K_hex)`

**Konsep:** Setiap baris citra diproses dengan keystream yang **BERGANTUNG PADA BARIS SEBELUMNYA**.

**⚠️ STATUS: PLACEHOLDER**

Implementasi saat ini adalah **SIMPLIFIED VERSION**. Versi final harus mengimplementasikan:

1. **Lookup Table (LT)** untuk setiap baris
2. **Update Parameter** per baris:
   ```
   LT(a,:) = LT(a,:) + 10^(-15) × 65536 × P'(a-1,:)
   ```
3. **Keystream Generation** dari LT(a,:)
4. **XOR Operation** row-wise

**Placeholder saat ini:**
```python
# Bangkitkan keystream per baris
keystream[a] = logistic_map(x0, r, W')
# XOR dengan baris
out[a, :] = Pprime[a, :] XOR keystream[a]
```

---

### **[TAHAP 4] GLOBAL CHAOS GENERATION**

#### **Fungsi:** `build_S1_S2_S3(K_hex, cfg, H_, W_)`

**Output:** Tiga sequence chaos:
- **S1:** Untuk confusion (length = H' + W')
- **S2:** Untuk diffusion (length = H' × W')
- **S3:** Untuk whitening (length = H' × W')

**ALGORITMA:**

1. **Generate Initial Value:**
   ```
   x0 = deterministik_dari_key(K_hex)  # x0 ∈ (0,1)
   ```

2. **Logistic Map Sequence:**
   ```
   x_{t+1} = r × x_t × (1 - x_t)
   ```
   Panjang: `T0 + (H'+W') + (H'×W') + (H'×W')`

3. **Buang Transient:**
   ```
   sequence_useful = sequence[T0:]  # Buang 500 iterasi pertama
   ```

4. **Kuantisasi:**
   ```
   RUMUS KUANTISASI:
   q_n = floor((1/Q) × 10^4 × (x_n mod 1))
   ```

5. **Konversi ke Byte:**
   ```
   byte = q mod 256
   ```

6. **Split Sequential:**
   ```
   S1 = byte[0 : H'+W']
   S2 = byte[H'+W' : H'+W'+H'×W']
   S3 = byte[H'+W'+H'×W' : ...]
   ```

> **💡 KEY POINT:** Semua sequence **DETERMINISTIK** dari kunci K, memastikan reproducibility.

---

### **[TAHAP 5] CONFUSION (Circular Shifts)**

#### **Fungsi:** `confusion_circular_shift(P_in, S1)`

**Tujuan:** **PERMUTASI POSISI PIKSEL** tanpa mengubah nilai.

**MEKANISME:**

1. **Split S1:**
   ```
   S1_row[i] = S1[i]      untuk i = 0..H'-1  (shift baris)
   S1_col[j] = S1[H'+j]   untuk j = 0..W'-1  (shift kolom)
   ```

2. **Operasi Shift:**
   ```
   STEP 1: Shift setiap KOLOM j sebesar S1_col[j]
   STEP 2: Shift setiap BARIS i sebesar S1_row[i]
   ```

3. **Circular Shift 1D:**
   ```python
   np.roll(array, shift)  # shift > 0 → geser kanan/bawah
   ```

**Contoh Visual:**
```
Original:     After Column Shift:    After Row Shift:
[1 2 3]       [3 1 2]                [2 3 1]
[4 5 6]   →   [6 4 5]           →    [5 6 4]
[7 8 9]       [9 7 8]                [8 9 7]
```

> **📌 PENTING:** Circular shift bersifat **REVERSIBLE** dengan shift negatif.

---

### **[TAHAP 6] DIFFUSION**

#### **Fungsi:** `diffusion_placeholder(P_in, S2)`

**Tujuan:** **MENYEBARKAN PERUBAHAN** satu piksel ke seluruh citra.

**⚠️ STATUS: PLACEHOLDER**

Implementasi saat ini menggunakan **DUAL SCAN (forward + backward)**:

1. **Forward Scan:**
   ```python
   y[0] = x[0] XOR k[0]
   for i = 1..N-1:
       y[i] = (x[i] XOR k[i]) XOR y[i-1]  # Chaining
   ```

2. **Backward Scan:**
   ```python
   z[N-1] = y[N-1] XOR k[N-1]
   for i = N-2..0:
       z[i] = (y[i] XOR k[i]) XOR z[i+1]  # Reverse chaining
   ```

**EFEK:**
- Perubahan 1 bit di `x[0]` mempengaruhi **SEMUA** `y[i]` dengan `i ≥ 0`
- Backward scan menambah difusi ke arah sebaliknya

**⚠️ TODO:** Ganti dengan diffusion multi-arah sesuai proposal (seperti pola "▲ ▼ ◄ ►").

---

### **[TAHAP 7] KEY WHITENING**

#### **Fungsi:** `key_whitening(P_in, S3)`

**Tujuan:** **MENAMBAHKAN LAPISAN KUNCI AKHIR** via XOR.

**RUMUS SEDERHANA:**
```
P_out[i,j] = P_in[i,j] XOR S3_mask[i,j]
```

**Implementasi:**
```python
mask = S3.reshape((H', W'))
P_out = P_in XOR mask
```

> **🔐 FUNGSI:** Menyembunyikan pola output diffusion dengan masking acak dari chaos, meningkatkan keamanan.

---

### **[TAHAP 8] INVERSE REARRANGEMENT**

#### **Fungsi:** `inverse_rearrange_to_image(Pout, meta)`

**Tujuan:** Mengubah matriks internal **kembali ke format citra** (grayscale/RGB).

**OPERASI INVERSE:**

1. **Grayscale:**
   ```
   C = Pout  (langsung)
   ```

2. **RGB (deinterleave columns):**
   ```python
   for j in range(W):
       R[:, j] = Pout[:, 3j + 0]
       G[:, j] = Pout[:, 3j + 1]
       B[:, j] = Pout[:, 3j + 2]
   
   C = stack([R, G, B], axis=2)  # Shape: (H, W, 3)
   ```

**OUTPUT:** Cipher-image `C` dengan format yang sama dengan input original.

---

## **🔄 ALUR EKSEKUSI LENGKAP**

### **Fungsi Master:** `encrypt_baseline(P, K_hex, cfg, return_debug)`

**Parameter Input:**
- `P`: Citra plaintext (numpy array, dtype=uint8)
- `K_hex`: Kunci 128-bit (string 32 karakter hex)
- `cfg`: Konfigurasi (optional, default=BaselineConfig())
- `return_debug`: Boolean untuk output intermediate (default=True)

**Return Value:**
- `C`: Citra terenkripsi
- `debug`: Dictionary berisi semua tahap intermediate

**Urutan Eksekusi:**

```python
# 1. KEY PROCESSING
Km_dec = split_key_128hex_to_subkeys(K_hex)
rm = rm_from_subkeys_oravec(Km_dec)

# 2. REARRANGEMENT
Pprime, meta = rearrange_image(P)

# 3. PLAINTEXT-RELATED
Ppr = plaintext_related_encrypt_placeholder(Pprime, rm, cfg, K_hex)

# 4. CHAOS GENERATION
S1, S2, S3 = build_S1_S2_S3(K_hex, cfg, H_, W_)

# 5. CONFUSION
Pconf = confusion_circular_shift(Ppr, S1)

# 6. DIFFUSION
Pdiff = diffusion_placeholder(Pconf, S2)

# 7. WHITENING
Pout = key_whitening(Pdiff, S3)

# 8. INVERSE REARRANGE
C = inverse_rearrange_to_image(Pout, meta)
```

---

## **⚙️ FUNGSI-FUNGSI UTILITAS**

### **1. Validasi & Konversi**

#### `_ensure_uint8(img)`
- Memastikan array adalah uint8
- Clip nilai ke range [0, 255]
- Essential untuk operasi XOR yang valid

---

### **2. Circular Shift Operations**

#### `_circular_shift_1d(arr, shift)`
**Konsep:** Rotasi elemen array secara circular
```
[1,2,3,4,5] shift=2 → [4,5,1,2,3]
```

#### `_circular_shift_rows(mat, shifts)`
**Operasi:** Shift setiap **BARIS i** sebesar `shifts[i]`
```python
for i in range(H):
    mat[i, :] = roll(mat[i, :], shifts[i])
```

#### `_circular_shift_cols(mat, shifts)`
**Operasi:** Shift setiap **KOLOM j** sebesar `shifts[j]`
```python
for j in range(W):
    mat[:, j] = roll(mat[:, j], shifts[j])
```

---

### **3. Chaos & Kuantisasi**

#### `logistic_map_sequence(x0, r, n)`
**Persamaan Fundamental:**
```
x_{t+1} = r × x_t × (1 - x_t)
```

**Parameter:**
- `x0 ∈ (0,1)`: Initial value
- `r ∈ [3.57, 4]`: Chaos parameter (semakin dekat 4 = semakin chaotic)
- `n`: Panjang sequence

**Karakteristik:**
- **NON-PERIODIC** untuk r dekat 4
- **SENSITIF** terhadap x0 (butterfly effect)
- **BOUNDED** dalam [0,1]

#### `quantize_sequence(seq, Q)`
**Rumus:**
```
q_n = floor((1/Q) × 10^4 × (x_n mod 1))
```

**Fungsi:** Mengubah float [0,1] → integer diskret untuk kriptografi

---

## **📊 DEBUG DICTIONARY**

Saat `return_debug=True`, fungsi `encrypt_baseline()` mengembalikan dictionary dengan keys:

| Key | Shape | Deskripsi |
|-----|-------|-----------|
| `Km_dec` | (8,) | Subkeys dari kunci K |
| `rm` | (8,) | Parameter chaos r_m |
| `Pprime` | (H', W') | Setelah rearrangement |
| `Ppr` | (H', W') | Setelah plaintext-related |
| `S1` | (H'+W',) | Confusion control |
| `S2` | (H'×W',) | Diffusion control |
| `S3` | (H'×W',) | Whitening mask |
| `Pconf` | (H', W') | Setelah confusion |
| `Pdiff` | (H', W') | Setelah diffusion |
| `Pout` | (H', W') | Setelah whitening |
| `C` | Original shape | Cipher-image final |

> **💡 USAGE:**Gunakan untuk analisis, debugging, dan visualisasi tahap-tahap enkripsi.

---

## **⚠️ BAGIAN YANG PERLU DIKUSTOMISASI**

### **1. PLAINTEXT-RELATED ENCRYPTION** (PRIORITAS TINGGI)

**File:** `plaintext_related_encrypt_placeholder()`

**Yang perlu diimplementasikan:**
- ✅ Lookup Table (LT) initialization
- ✅ Update parameter per baris: `LT(a,:) = LT(a,:) + 10^(-15) × 65536 × P'(a-1,:)`
- ✅ Keystream generation dari LT(a,:)
- ✅ XOR operation sesuai persamaan (2.5) proposal

---

### **2. DIFFUSION** (PRIORITAS TINGGI)

**File:** `diffusion_placeholder()`

**Yang perlu diimplementasikan:**
- ✅ Multi-directional diffusion (▲ ▼ ◄ ►)
- ✅ Chaining mechanism sesuai proposal
- ✅ Invertible operation untuk dekripsi nanti

---

### **3. GLOBAL CHAOS GENERATION** (PRIORITAS SEDANG)

**File:** `build_S1_S2_S3()`

**Verifikasi:**
- ✅ Cara pembentukan sequence global
- ✅ Pemotongan/concatenation S1, S2, S3
- ✅ Pastikan deterministik dan reproducible

---

## **🧪 TESTING & VALIDATION**

### **Smoke Test (Built-in)**

File menyediakan contoh test di bagian `if __name__ == "__main__"`:

```python
# Generate dummy image 16×16 grayscale
P = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)

# Kunci dummy
K = "00112233445566778899aabbccddeeff"

# Encrypt
C, dbg = encrypt_baseline(P, K, return_debug=True)

print("Plain shape:", P.shape)      # (16, 16)
print("Cipher shape:", C.shape)     # (16, 16)
print("Stages:", list(dbg.keys()))  # Semua tahap intermediate
```

### **Test yang Disarankan:**

1. **Determinism Test:**
   ```python
   C1, _ = encrypt_baseline(P, K)
   C2, _ = encrypt_baseline(P, K)
   assert np.array_equal(C1, C2)  # Harus identik
   ```

2. **Key Sensitivity Test:**
   ```python
   K1 = "00112233445566778899aabbccddeeff"
   K2 = "00112233445566778899aabbccddeeFF"  # Beda 1 bit
   C1, _ = encrypt_baseline(P, K1)
   C2, _ = encrypt_baseline(P, K2)
   assert not np.array_equal(C1, C2)  # Harus berbeda signifikan
   ```

3. **RGB Test:**
   ```python
   P_rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
   C_rgb, _ = encrypt_baseline(P_rgb, K)
   assert C_rgb.shape == (32, 32, 3)
   ```

---

## **📚 REFERENSI RUMUS PENTING**

### **1. Parameter Chaos dari Kunci**
```
r_m = 4 - 10^(-15) × ((9-m) × 256 × 65536 - K_m)
```

### **2. Logistic Map**
```
x_{t+1} = r × x_t × (1 - x_t)
```

### **3. Kuantisasi**
```
q_n = floor((1/Q) × 10^4 × (x_n mod 1))
```

### **4. Key Whitening**
```
P_out = P_in XOR S3_mask
```

### **5. Diffusion Chaining (Placeholder)**
```
Forward:  y[i] = (x[i] XOR k[i]) XOR y[i-1]
Backward: z[i] = (y[i] XOR k[i]) XOR z[i+1]
```

---

## **🎯 KESIMPULAN**

### **Kelebihan Implementasi:**
✅ Modular dan mudah di-maintain  
✅ Type hints untuk clarity  
✅ Extensive comments di bagian penting  
✅ Debug output untuk analisis  
✅ Deterministik dan reproducible  

### **Yang Perlu Dilengkapi:**
⚠️ Implementasi final plaintext-related (LT mechanism)  
⚠️ Implementasi final diffusion multi-arah  
⚠️ Validasi kesesuaian dengan proposal asli  
⚠️ Implementasi dekripsi (inverse pipeline)  

---

## **📞 NEXT STEPS**

1. **Verifikasi** rumus-rumus dengan proposal/paper asli
2. **Implementasikan** plaintext-related encryption dengan benar
3. **Implementasikan** diffusion multi-directional
4. **Test** dengan berbagai ukuran citra dan kunci
5. **Develop** inverse pipeline untuk dekripsi
6. **Validasi** dengan metrik keamanan (NPCR, UACI, entropy, dll)

---

**_Dokumentasi dibuat untuk memudahkan pemahaman pipeline enkripsi chaos-based. Update sesuai perkembangan implementasi._**
