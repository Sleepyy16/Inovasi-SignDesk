# SignDesk (nama sementara) — Penerjemah Alfabet BISINDO untuk Layanan Loket

Alat bantu komunikasi berbasis kamera yang membantu petugas loket (pendaftaran, administrasi, layanan publik) berkomunikasi dengan pengguna bahasa isyarat BISINDO. **Fase 1 (dokumen ini): pengenalan alfabet A–Z.**

> Status: perencanaan. Nama proyek, angka target, dan jadwal di bawah adalah titik awal. Sesuaikan setelah baseline pertama jadi.

---

## 1. Masalah dan Target Pengguna

Banyak layanan tatap muka (pendaftaran rumah sakit, administrasi kampus, loket layanan publik) tidak punya petugas yang menguasai bahasa isyarat. Akibatnya penyandang Tuli harus menulis di kertas, meminta bantuan pendamping, atau menyerah di tengah proses.

| Pengguna | Kebutuhan |
|---|---|
| **Pengunjung Tuli** (pengguna BISINDO) | Menyampaikan informasi dasar, misalnya mengeja nama, tanpa harus menulis atau membawa pendamping |
| **Petugas loket** (tidak paham isyarat) | Melihat apa yang disampaikan pengunjung sebagai teks, dan membalas dengan cara yang bisa dipahami |

**Konteks fokus:** loket pendaftaran dan layanan yang banyak berinteraksi dengan publik.

## 2. Cakupan

### Termasuk dalam MVP (Fase 1)
- Pengenalan alfabet BISINDO A–Z dari webcam secara real-time (Isyarat → Teks).
- Penyusun teks: huruf yang terdeteksi stabil masuk ke kolom teks (untuk mengeja nama dll).
- Mode Loket: tampilan dua sisi, **Pengunjung** (kamera + teks hasil isyarat) dan **Petugas** (mengetik balasan).
- Teks → Isyarat versi sederhana: teks petugas ditampilkan sebagai urutan gambar huruf isyarat (fingerspelling).
- Evaluasi model dengan pemisahan per penutur.

### Di luar cakupan (jangan dikerjakan dulu)
- Kosakata/kata isyarat dan kalimat kontinu (Fase 2, roadmap).
- Avatar 3D atau terjemahan teks → isyarat yang natural.
- Aplikasi mobile native, akun pengguna, database pengunjung.
- Penyimpanan video atau data pribadi.

### Catatan jujur tentang alfabet
- Orang Tuli jarang berkomunikasi dengan mengeja huruf per huruf. Alfabet berguna untuk **nama, singkatan, dan istilah yang tidak punya isyarat**, bukan untuk percakapan penuh. Karena itu alfabet adalah **fondasi teknis**, dan nilai dampak yang lebih besar ada di Fase 2 (kosakata layanan).
- **Banyak huruf BISINDO memakai dua tangan**, sehingga sistem harus mampu membaca dua tangan sekaligus.
- Beberapa huruf (kemungkinan termasuk J dan Z) melibatkan gerakan. Di MVP keduanya diperlakukan sebagai pose statis dan ditandai **eksperimental**. **Verifikasi semua huruf dengan referensi resmi atau penutur BISINDO.**

---

## 3. Arsitektur

### 3.1 Alur Isyarat → Teks

```text
Webcam
  │  (frame video)
  ▼
[1] Deteksi tangan — MediaPipe Hands (maks. 2 tangan, 21 titik per tangan)
  │  (landmark x, y, z)
  ▼
[2] Ekstraksi fitur — normalisasi posisi & skala, urutkan tangan, padding jika 1 tangan
  │  (vektor fitur ±127 angka)
  ▼
[3] Klasifikasi — model ML (Random Forest / MLP) → huruf + skor keyakinan
  │
  ▼
[4] Penstabil — voting beberapa frame terakhir + ambang keyakinan + tahan ±1 detik
  │  (huruf terkonfirmasi)
  ▼
[5] Penyusun teks — tambah huruf, Spasi, Hapus, Reset
  │
  ▼
[6] Antarmuka — tampilan Pengunjung / Petugas (Streamlit)
```

### 3.2 Alur Teks → Isyarat (versi MVP)

```text
Petugas mengetik teks
  ▼
Normalisasi (huruf besar, buang karakter tak didukung)
  ▼
Cocokkan tiap huruf ke gambar isyarat (folder assets/alfabet/)
  ▼
Tampilkan urutan gambar (grid atau slideshow)
```

### 3.3 Alur Pelatihan Model (dikerjakan offline)

```text
Sumber data A: dataset gambar publik ──► extract_from_images.py ─┐
                                                                  ├─► CSV landmark ──► train.py ──► model.joblib
Sumber data B: rekaman sendiri (webcam) ──► collect_data.py ─────┘                     │
                                                                                       ▼
                                                                                evaluate.py
                                                                       (akurasi per penutur, confusion matrix)
```

### 3.4 Kenapa landmark, bukan gambar mentah?

- Model jauh lebih **ringan** dan cepat, cukup untuk laptop biasa.
- Lebih **tahan terhadap latar, pakaian, dan pencahayaan**, karena yang dipelajari adalah bentuk tangan, bukan pikselnya.
- Cocok untuk **pemula**: tidak perlu GPU dan tidak perlu melatih jaringan saraf besar.

### 3.5 Detail ekstraksi fitur (penting, mudah salah)

1. Ambil landmark semua tangan yang terdeteksi (1 atau 2 tangan).
2. **Urutkan tangan berdasarkan posisi x pergelangan tangan** (kiri ke kanan di layar), **bukan berdasarkan label "Left/Right"** dari MediaPipe. Label itu bisa tertukar saat gambar dicerminkan.
3. **Normalisasi bersama** untuk semua titik: kurangi titik pusat, lalu bagi dengan skala (jarak maksimum). Normalisasi bersama menjaga hubungan posisi antar dua tangan, yang penting untuk huruf dua tangan.
4. Jika hanya 1 tangan: isi slot tangan kedua dengan nol dan tambahkan fitur penanda jumlah tangan.
5. Gunakan **fungsi ekstraksi yang sama persis** untuk pelatihan dan real-time. Perbedaan kecil antar keduanya adalah sumber bug paling umum ("akurat saat training, ngawur saat dipakai").

### 3.6 Penstabil prediksi (supaya teks tidak "berantakan")

- Simpan N prediksi terakhir (misalnya 10 frame), lalu ambil yang paling sering muncul.
- Abaikan jika skor keyakinan di bawah ambang (misalnya 0.7).
- Huruf baru dimasukkan ke teks hanya jika **stabil sekitar 1 detik**, dan harus ada jeda (tangan turun/berubah) sebelum huruf yang sama bisa dimasukkan lagi.
- Semua angka ini disimpan di `config.py` agar mudah diatur.

---

## 4. Teknologi

| Komponen | Pilihan | Alasan |
|---|---|---|
| Bahasa | Python 3.10/3.11 | Ekosistem ML paling mudah untuk pemula (cek kompatibilitas MediaPipe sebelum memakai versi Python yang lebih baru) |
| Deteksi tangan | MediaPipe Hands | Gratis, cepat, berjalan di CPU |
| Kamera/gambar | OpenCV | Standar untuk pengolahan video |
| Model | scikit-learn (Random Forest, MLP) | Mudah, cepat dilatih, tanpa GPU |
| Aplikasi | Streamlit + streamlit-webrtc | Antarmuka web dengan webcam tanpa menulis frontend |
| Data | pandas + CSV | Sederhana dan mudah diperiksa |
| Visualisasi | matplotlib | Confusion matrix dan grafik |

Kunci (pin) versi semua pustaka di `requirements.txt` supaya semua anggota tim punya lingkungan yang sama.

## 5. Struktur Repo

```text
signdesk/
├── README.md
├── requirements.txt
├── config.py                  # semua pengaturan (ambang, N frame, path)
├── data/
│   ├── raw_images/            # dataset gambar publik (tidak di-commit)
│   ├── landmarks/             # CSV hasil ekstraksi
│   └── README.md              # sumber, lisensi, cara unduh
├── models/                    # model terlatih (.joblib)
├── assets/
│   └── alfabet/               # gambar isyarat A–Z untuk Teks → Isyarat
├── src/
│   ├── features.py            # ekstraksi & normalisasi landmark (dipakai bersama)
│   ├── collect_data.py        # rekam data sendiri lewat webcam
│   ├── extract_from_images.py # landmark dari dataset gambar
│   ├── train.py               # latih model
│   ├── evaluate.py            # evaluasi per penutur + confusion matrix
│   ├── stabilizer.py          # penstabil prediksi
│   ├── text_to_sign.py        # teks → urutan gambar isyarat
│   └── realtime.py            # demo OpenCV sederhana
├── app/
│   └── app.py                 # aplikasi Streamlit (Mode Loket)
└── reports/                   # hasil evaluasi, grafik, catatan uji pengguna
```

## 6. Cara Menjalankan (rencana)

```bash
# 1. Buat lingkungan virtual
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Pasang pustaka
pip install -r requirements.txt

# 3. Siapkan data (lihat data/README.md), lalu ekstrak landmark
python src/extract_from_images.py

# 4. (Disarankan) tambah data sendiri
python src/collect_data.py

# 5. Latih dan evaluasi
python src/train.py
python src/evaluate.py

# 6. Coba real-time sederhana
python src/realtime.py

# 7. Jalankan aplikasi
streamlit run app/app.py
```

## 7. Dataset

Sumber publik yang bisa diperiksa (**cek lisensi dan syarat sitasi masing-masing sebelum memakai**, lalu catat di `data/README.md`):

- Dataset alfabet BISINDO di **Kaggle** (judul: *Bahasa Isyarat Indonesia (BISINDO) Alphabets*, pengunggah: achmadnoer).
- **BISINDO DATASET** di Mendeley Data: gambar gestur tangan untuk 26 huruf, ditujukan untuk riset computer vision dan MediaPipe Hands.
- **BISINDO SIGN LANGUAGE RECOGNITION** di Mendeley Data: gambar gestur alfabet dan beberapa kelas gestur umum.

**Peringatan:** dataset publik biasanya berisi sedikit penutur, sehingga model bisa terlihat akurat tapi gagal pada orang lain. Karena itu:

1. **Rekam data sendiri** dari banyak orang (idealnya penutur BISINDO asli), dengan variasi jarak, cahaya, dan latar.
2. Catat `person_id` di setiap sampel.
3. **Pisahkan data latih/uji berdasarkan orang**, bukan acak per gambar.
4. Dapatkan **persetujuan tertulis** dari setiap orang yang direkam, dan jelaskan data apa yang disimpan (hanya angka landmark, bukan wajah).

## 8. Evaluasi

| Metrik | Cara ukur | Target awal (sesuaikan) |
|---|---|---|
| Akurasi pada penutur yang tidak ada di data latih | Split per `person_id` | ≥ 85% |
| Akurasi per huruf | Laporan klasifikasi | Tidak ada huruf < 70% |
| Huruf yang sering tertukar | Confusion matrix | Dicatat dan dianalisis |
| Kecepatan | Waktu per frame di laptop biasa | < 200 ms |
| Uji pengguna | Skenario "mengeja nama di loket" dengan penutur BISINDO | Tingkat keberhasilan dan skor kepuasan |

Laporkan hasil **apa adanya**, termasuk kelemahan. Juri lebih menghargai angka jujur beserta analisisnya daripada angka tinggi yang tidak bisa dipertanggungjawabkan.

## 9. Etika, Privasi, dan Keterbatasan

- **Tidak menggantikan juru bahasa isyarat.** Posisikan sebagai alat bantu komunikasi dasar. Jangan mengklaim sebaliknya di proposal atau presentasi.
- **Libatkan komunitas Tuli sejak awal**: validasi isyarat, jadi penutur data, dan beri masukan desain. Contoh mitra: GERKATIN, SLB terdekat, komunitas BISINDO kampus.
- **Privasi:** video diproses di perangkat dan **tidak disimpan**. Aplikasi tidak menyimpan nama atau data pengunjung. Jelaskan ini kepada pengguna di layar.
- **Keterbatasan yang harus disebut:** hanya alfabet, sensitif terhadap cahaya dan sudut kamera, huruf bergerak (J, Z) belum akurat, dan belum menangani ekspresi wajah/tata bahasa isyarat.

## 10. Pemetaan ke Kriteria Lomba

| Kriteria | Argumen |
|---|---|
| **SDG 4** Pendidikan berkualitas | Akses komunikasi dan belajar yang inklusif |
| **SDG 8** Pekerjaan layak dan pertumbuhan ekonomi | Layanan publik dan tempat kerja yang lebih setara bagi penyandang disabilitas |
| **SDG 9** Industri, inovasi, infrastruktur | Teknologi asistif berbasis computer vision |
| **Asta Cita 4** | Penguatan SDM dan teknologi, termasuk inklusi bagi penyandang disabilitas |

Di proposal, tulis eksplisit: *"Proyek ini mendukung SDG 4, 8, dan 9 serta Asta Cita 4 karena ..."* dan sertakan angka dampak dari uji pengguna.

## 11. Pembagian Peran (untuk tim 3–5 orang)

| Peran | Tugas utama |
|---|---|
| **Data & Komunitas** | Hubungi komunitas/SLB, kumpulkan data, urus persetujuan, dokumentasi sumber data |
| **Model** | Ekstraksi fitur, pelatihan, evaluasi, analisis kesalahan |
| **Aplikasi** | Real-time, penstabil, Streamlit, Teks → Isyarat |
| **Dokumentasi & Pitch** | README, laporan evaluasi, proposal, video demo, uji pengguna |

Satu orang boleh memegang dua peran. Yang penting setiap peran punya penanggung jawab.

## 12. Roadmap

| Minggu | Target | Hasil yang bisa dicek |
|---|---|---|
| 1 | Setup lingkungan, hubungi komunitas, unduh dataset, `features.py`, `extract_from_images.py` | CSV landmark terbentuk, laporan gambar yang gagal terdeteksi |
| 2 | `collect_data.py`, kumpulkan data sendiri, `train.py`, `evaluate.py` | Akurasi baseline + confusion matrix |
| 3 | `stabilizer.py`, `realtime.py`, mulai `app.py` | Bisa mengeja sebuah kata lewat webcam |
| 4 | Mode Loket, Teks → Isyarat, uji pengguna, perbaikan | Video demo + data uji pengguna |
| Fase 2 | Kosakata layanan (nama, "tolong", "daftar", "dokter", dll.) dengan model sekuens | Roadmap di proposal |

## 13. Definition of Done (MVP)

- [ ] Aplikasi berjalan dengan `streamlit run app/app.py` di laptop anggota tim lain (bukan hanya di laptop pembuat).
- [ ] Bisa mengeja minimal satu nama lengkap lewat webcam dengan koreksi (Spasi/Hapus).
- [ ] Teks petugas tampil sebagai urutan gambar isyarat.
- [ ] Evaluasi per penutur sudah dilaporkan beserta confusion matrix.
- [ ] Minimal satu sesi uji dengan penutur BISINDO asli sudah dicatat.
- [ ] README ini sudah diperbarui sesuai kenyataan (angka, struktur, batasan).

## 14. Risiko dan Mitigasi

| Risiko | Mitigasi |
|---|---|
| Model akurat di data uji tapi gagal di dunia nyata | Split per penutur, tambah data sendiri, uji di berbagai kondisi |
| Label tangan kiri/kanan tertukar | Urutkan tangan berdasarkan posisi x (lihat 3.5) |
| Fitur training dan real-time berbeda | Satu fungsi ekstraksi bersama di `features.py` |
| Huruf bergerak (J, Z) tidak terdeteksi | Tandai eksperimental, dokumentasikan, jadikan bahan Fase 2 |
| Kesulitan mengakses komunitas Tuli | Mulai dari SLB atau organisasi disabilitas kampus sejak minggu 1 |
| Versi pustaka bentrok antar anggota | Pin versi di `requirements.txt`, pakai virtual environment |
| Lingkup membengkak | Pegang daftar "di luar cakupan" di bagian 2 |

## 15. Lisensi dan Atribusi

Catat lisensi kode proyek ini (misalnya MIT) dan **semua dataset yang dipakai beserta sitasinya** di `data/README.md`.
