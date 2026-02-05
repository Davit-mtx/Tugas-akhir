Plan TA bab 4, 5, 6



**Bab 4 Implementasi dan hasil eksperimen**

**Bab 5 Analisis dan hasil pembahasan**

**Bab 6 Kesimpulan dan saran**



**Week 1 - Setup \& desain eksperimen final**

* create structure folder
* fix random seed untuk sampling dan HO
* kontrak eksperimen bisa menyesuaikan rencana atau bikin ulang sesuai kemampuan device



**Week 2 - Akusisi KADID-10K + stratifikasi DMOS (low 10/medium 10/high 10)**

* load all dataset
* stratified random sampling citra berdasarkan keterangan
* output samples 30 citra terstratifikasi



**Week 3 - Implementasi baseline enkripsi-dekripsi**

* implement seluruh pipeline enkripsi dan dekripsi sampai exact match Plaintext=Plaintext(hasil dekripsi)
* semua operasi deterministik
* output: beberapa sample lolos uji coba (kalau bisa semua sampel citra lolos)



**Week 4 - Hardering deterministik + validasi tahap (debuggable/memperbaiki bug secara real time)**

* debug code yang berpotensi vital pada saat deterministik atau invertible
* pastikan urutan program sudah sesuai dengan yang di rencanakan pada bab 3



**Week 5 - Implement metrik dasar**

* entropi per kanal + rata-rata
* korelasi secara horizontal/vertikal/diagonal per kanal juga
* waktu enkripsi dan dekripsi + rata-rata N repetisi yang di hasilkan
* output file data baseline metrik yang berisikan 30 sampel data citra



**Week 6 - integrasi fitness F(x) + penalty**

* implementasi fungsi fitness dengan penalty
* dengan menggabungkan penalty dekripsi dan penalty chaos
* implement operator repair menggunakan transient T0 dan kuantisasi Q
* Output: fitness dijalankan untuk tiap citranya membedakan kadidat baik dan buruknya



**Week 7 - implement HO (core) + greedy selection**

* implement 3 fase HO + greedy acceptance
* logging per-iterasi : bet fitness, best x, jumlah kadidat kena penalty 
* output: HO bisa jalan pada 1 citra 



**Week 8 - tuning HO agar runtime masuk akal**

* profilling bottleneck (baiasanya diffusion+ korelasi)
* optimasi: vectorization, caching bagian yang bisa dicache, batasi perhitungan korelasi (jumlah pasangan tetap)
* putuskan "mode cepat", mis: korelasi pakai subset pasangan piksel untuk fitness, full korelasi untuk evaluasi akhir
* output: runtime HO per-citra "masuk akal"



**Week 9 - uji subset eksperiment berdasarkan tipe stratifikasi DMOS**

* jalankan baseline Xo VS X parameter optimal yang dihasilkan pada HO
* simpan parameter optimal X, hasil perhitungan metrik evaluasi secara keseluruhan sesuai dengan kebutuhan analisis evaluasi
* output: laporan atau mini report 



**Week 10 - jalankan eksperimen penuh 30 citra (baseline dan HO optimization)**

* run seluruh citra baik secara baseline maupun dengan optimasi HO
* output : 1. results baseline.csv, 2. reseult HO.csv



**Week 11 - perhitungan metrik evaluasi secara keseluruhan**

* hitung metrik evaluasi baik yang digunakan dalam enkripsi, dekripsi, uji kunci sensitivity
* output file metrik evaluasi dengan kolom nama image beserta bahan yang digunakan dalam perhitungan, tersedia di semua kolom sehingga nanti bisa di bandingkan



**Week 12 - penulisan laporan bab 4, 5, 6**

* struktur penulisan bab dan subbab tiap babnya yang benar secara urutan dan penggunakan teks penamaannya juga
* penambahan lampiran pendukung bab" sebelumnya terutama bab 4
* output : final buku TA yang siap dikumpulkan sebagai hasil akhir



**Week 13 - pembuatan ppt sidang**

* structure slide yang tersistem dan story telling yang kuat sehingga mudah dipahami
* output: ppt slide presentasi TA



**Week 14 - pembuatan poster seminar hasil**

* penetapan point apa saja yang perlu ditampilkan pada posters semhas
* penetapan layout yang digunakan sehingga rapi dan enak dilihat 
* output: soft file poster semhas yang siap untuk dipamerkan



