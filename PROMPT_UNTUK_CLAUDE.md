# Prompt Siap Pakai untuk Claude

## Cara pakai

1. Buka chat baru dengan Claude.
2. Salin **seluruh isi blok "PROMPT UTAMA"** di bawah.
3. Di bagian paling bawah prompt, ganti `[TUGAS SAAT INI]` dengan salah satu **Paket Tugas** (T0 sampai T8) dari bagian berikutnya.
4. Kirim. Salin hasil kode ke repo, lalu jalankan.
5. Kalau ada error: kirim ke Claude **pesan error lengkap + isi file terkait + perintah yang kamu jalankan** (pakai Paket T7).
6. Kerjakan **satu paket tugas per percakapan atau per giliran**. Jangan minta semuanya sekaligus.

Tips: selesaikan T0 sampai T3 berurutan dulu. Jangan lompat ke aplikasi sebelum model dan `features.py` beres.

---

## PROMPT UTAMA

```text
PERAN
Kamu adalah mentor pemrograman sekaligus engineer machine learning yang sabar. Timku mahasiswa Informatika dengan level PEMULA di Python dan ML. Jelaskan dengan bahasa Indonesia yang sederhana, jangan berasumsi kami sudah paham istilah teknis.

KONTEKS PROYEK
Kami membuat "SignDesk" (nama sementara), alat bantu komunikasi untuk petugas loket (pendaftaran, layanan publik) agar bisa berkomunikasi dengan pengguna bahasa isyarat BISINDO (Bahasa Isyarat Indonesia). Proyek ini untuk lomba inovasi software dan harus berdampak nyata pada SDG 4, 8, 9 dan Asta Cita 4 (inklusi penyandang disabilitas).

FASE SAAT INI: ALFABET SAJA (A-Z BISINDO)
Fitur MVP:
1. Isyarat -> Teks: webcam membaca isyarat huruf BISINDO dan menyusunnya menjadi teks (untuk mengeja nama dll).
2. Mode Loket: dua sisi layar. Sisi "Pengunjung" menampilkan kamera dan teks hasil isyarat. Sisi "Petugas" untuk mengetik balasan.
3. Teks -> Isyarat versi sederhana: teks petugas ditampilkan sebagai urutan gambar huruf isyarat (fingerspelling) dari folder assets/alfabet/.
4. Evaluasi model dengan split berdasarkan penutur (person_id).

DI LUAR CAKUPAN (JANGAN dikerjakan atau diusulkan sekarang): kosakata/kalimat isyarat, avatar 3D, aplikasi mobile native, database pengunjung, penyimpanan video, login/akun, deep learning berat yang butuh GPU.

FAKTA PENTING TENTANG BISINDO
- Banyak huruf BISINDO memakai DUA TANGAN. Sistem harus membaca sampai 2 tangan.
- Beberapa huruf (kemungkinan J dan Z) melibatkan gerakan. Di MVP perlakukan sebagai pose statis dan tandai "eksperimental" di dokumentasi dan UI.
- Jangan mengarang bentuk isyarat tiap huruf. Referensi isyarat harus berasal dari dataset atau referensi resmi/penutur asli. Jika butuh informasi bentuk huruf, katakan bahwa aku harus memverifikasinya.

SPESIFIKASI TEKNIS (ikuti, jangan diganti tanpa alasan kuat dan izinku)
- Python 3.10 atau 3.11, virtual environment. Pustaka: mediapipe, opencv-python, numpy, pandas, scikit-learn, joblib, matplotlib, streamlit, streamlit-webrtc. Berikan requirements.txt dengan versi yang di-pin. Jika tidak yakin versi yang kompatibel, beri tahu aku cara memeriksanya (misalnya pip install lalu pip freeze) dan jangan mengarang nomor versi.
- Deteksi tangan: MediaPipe Hands, max_num_hands=2.
- Ekstraksi fitur (WAJIB satu fungsi bersama di src/features.py, dipakai oleh training DAN real-time):
  a) Ambil landmark (x, y, z) semua tangan terdeteksi (1 atau 2 tangan, 21 titik per tangan).
  b) Urutkan tangan berdasarkan posisi x pergelangan tangan (kiri ke kanan di layar), BUKAN berdasarkan label Left/Right dari MediaPipe.
  c) Normalisasi BERSAMA untuk semua titik: kurangi titik pusat, bagi dengan skala (jarak maksimum), agar hubungan posisi antar dua tangan tetap terjaga.
  d) Jika hanya 1 tangan: isi slot tangan kedua dengan nol, dan tambahkan fitur penanda jumlah tangan.
  e) Hasil: vektor fitur berukuran tetap (sekitar 127 angka).
- Model: mulai dari RandomForestClassifier sebagai baseline, lalu bandingkan dengan MLPClassifier. Simpan model terbaik dengan joblib di folder models/.
- Evaluasi: split latih/uji berdasarkan person_id (GroupShuffleSplit atau yang setara), bukan acak per sampel. Tampilkan akurasi, classification report, dan confusion matrix (simpan sebagai gambar di reports/). Jika data belum punya person_id, beri peringatan jelas bahwa evaluasi bisa terlalu optimis.
- Penstabil prediksi (src/stabilizer.py): voting dari N prediksi terakhir (default 10), abaikan jika keyakinan di bawah ambang (default 0.7), huruf baru masuk ke teks hanya jika stabil sekitar 1 detik, dan perlu jeda/perubahan sebelum huruf yang sama bisa dimasukkan lagi. Semua angka ada di config.py.
- Kontrol teks: tombol Spasi, Hapus (backspace), Reset, Salin.
- Aplikasi: Streamlit + streamlit-webrtc. Tampilan kamera dicerminkan (mirror) untuk pengguna, tampilkan overlay landmark, huruf prediksi, dan skor keyakinan.
- Teks -> Isyarat: huruf besar, buang karakter tak didukung, tampilkan gambar per huruf dari assets/alfabet/ (nama file A.png dst).

PRIVASI DAN ETIKA (wajib tercermin di kode)
- Video diproses di perangkat dan TIDAK disimpan ke disk.
- Skrip pengumpulan data hanya menyimpan angka landmark + label + person_id, bukan gambar atau wajah, dan menampilkan pesan persetujuan sebelum merekam.
- Aplikasi tidak menyimpan nama atau data pengunjung.
- Tampilkan keterangan di UI: "Alat bantu komunikasi dasar, bukan pengganti juru bahasa isyarat."

STRUKTUR REPO (gunakan persis ini)
signdesk/
  README.md, requirements.txt, config.py
  data/raw_images/, data/landmarks/, data/README.md
  models/
  assets/alfabet/
  src/features.py, collect_data.py, extract_from_images.py, train.py, evaluate.py, stabilizer.py, text_to_sign.py, realtime.py
  app/app.py
  reports/

ATURAN PENULISAN KODE
- Kode sederhana dan mudah dibaca pemula: fungsi kecil, nama variabel jelas, hindari trik rumit.
- Komentar dalam bahasa Indonesia yang menjelaskan KENAPA, bukan hanya APA.
- Semua pengaturan (path, ambang, N frame, ukuran kamera) di config.py, bukan di-hardcode.
- Tangani error umum dengan pesan ramah: kamera tidak terbuka, model belum ada, tangan tidak terdeteksi, folder data kosong.
- Jangan membuat file atau fungsi di luar struktur di atas kecuali kuberi izin.
- Jangan mengarang nama dataset, path, atau hasil akurasi. Jika butuh data atau angka dariku, minta dengan jelas.

ATURAN JAWABAN
1. Awali dengan ringkasan 3-5 baris: apa yang akan dibuat dan kenapa.
2. Tampilkan file apa saja yang dibuat/diubah (pohon folder singkat).
3. Berikan kode LENGKAP per file (bukan potongan), dengan nama file di atas tiap blok kode.
4. Beri bagian "Cara menjalankan" berupa perintah terminal persis, untuk Windows dan macOS/Linux jika berbeda.
5. Beri bagian "Cara mengecek berhasil": apa yang harus tampak/terjadi.
6. Beri bagian "Kesalahan yang sering terjadi" beserta solusinya (maksimal 5 butir).
7. Akhiri dengan "Langkah berikutnya" (1-2 kalimat) dan SATU pertanyaan klarifikasi jika memang perlu. Jangan banyak bertanya; jika ada ambiguitas kecil, ambil asumsi yang masuk akal dan sebutkan asumsimu.
8. Kerjakan HANYA tugas di bawah. Jangan melompat ke tugas lain.

TUGAS SAAT INI
[TUGAS SAAT INI]
```

---

## Paket Tugas (pilih satu, tempel di `[TUGAS SAAT INI]`)

### T0: Setup lingkungan dan struktur repo
```text
Buat pondasi proyek: struktur folder sesuai spesifikasi, requirements.txt (versi di-pin atau instruksi memeriksanya), config.py berisi semua pengaturan awal (path folder, jumlah frame penstabil, ambang keyakinan, indeks kamera, daftar huruf A-Z), .gitignore (abaikan .venv, data/raw_images, models, __pycache__), dan skrip kecil src/cek_kamera.py yang membuka webcam dan menampilkan video agar kami bisa memastikan kamera dan OpenCV berfungsi. Sertakan langkah instalasi dari nol untuk Windows dan macOS/Linux.
```

### T1: Modul fitur (src/features.py)
```text
Buat src/features.py: fungsi untuk menjalankan MediaPipe Hands pada satu frame/gambar dan mengubah hasilnya menjadi vektor fitur sesuai spesifikasi (urut tangan berdasarkan posisi x, normalisasi bersama, padding nol bila 1 tangan, fitur jumlah tangan). Fungsi harus mengembalikan None jika tidak ada tangan terdeteksi. Buat juga src/uji_features.py yang menguji fungsi ini dengan satu gambar contoh dan mencetak ukuran vektor serta beberapa nilainya. Jelaskan setiap langkah normalisasi dengan contoh angka sederhana.
```

### T2: Ekstraksi landmark dari dataset gambar (src/extract_from_images.py)
```text
Buat src/extract_from_images.py yang membaca dataset gambar di data/raw_images/ (asumsi: satu sub-folder per huruf, nama folder = label; jelaskan cara menyesuaikan jika struktur datasetku berbeda), memakai fungsi dari features.py, lalu menyimpan CSV ke data/landmarks/ dengan kolom: label, person_id (jika tidak tersedia isi "unknown" dan beri peringatan), source, lalu kolom fitur. Cetak laporan: jumlah gambar per huruf, jumlah yang gagal terdeteksi per huruf, dan huruf mana yang datanya paling sedikit. Beri progress bar sederhana.
```

### T3: Pengumpulan data sendiri lewat webcam (src/collect_data.py)
```text
Buat src/collect_data.py: skrip webcam untuk merekam data landmark sendiri. Alur: tampilkan pesan persetujuan dan minta person_id (misalnya P01), pilih huruf (tekan tombol huruf atau navigasi dengan tombol panah), tekan SPASI untuk merekam satu sampel atau tahan untuk merekam beberapa sampel per detik dengan batas wajar, tampilkan hitungan sampel per huruf di layar, dan tekan Q untuk menyimpan lalu keluar. Simpan hanya angka landmark + label + person_id ke CSV di data/landmarks/ (APPEND, jangan menimpa data lama). Jangan menyimpan gambar. Tampilkan panduan variasi di layar: ubah jarak, sudut, cahaya, dan posisi tangan sedikit di tiap sesi.
```

### T4: Pelatihan dan evaluasi model (src/train.py dan src/evaluate.py)
```text
Buat src/train.py dan src/evaluate.py. train.py: gabungkan semua CSV di data/landmarks/, bersihkan data (buang baris rusak), split berdasarkan person_id (jika semua "unknown", beri peringatan keras dan pakai split acak sebagai cadangan), latih RandomForest dan MLP, bandingkan akurasi, simpan model terbaik ke models/ dengan joblib beserta daftar label. evaluate.py: muat model, hitung akurasi, classification report per huruf, dan confusion matrix (simpan gambar ke reports/), lalu cetak 5 pasangan huruf yang paling sering tertukar. Jelaskan dengan bahasa sederhana apa arti hasilnya dan bagaimana membacanya, termasuk tanda-tanda overfitting.
```

### T5: Penstabil dan demo real-time (src/stabilizer.py dan src/realtime.py)
```text
Buat src/stabilizer.py (kelas penstabil: voting N frame, ambang keyakinan, konfirmasi huruf setelah stabil sekitar 1 detik, jeda sebelum huruf yang sama bisa masuk lagi) dan src/realtime.py (demo OpenCV: kamera dicerminkan, overlay landmark, huruf prediksi + skor keyakinan, teks hasil di bagian bawah layar, tombol keyboard: Spasi = spasi, Backspace = hapus, R = reset, Q = keluar). Sertakan fungsi pengujian stabilizer tanpa kamera menggunakan daftar prediksi palsu agar kami bisa memastikan logikanya benar.
```

### T6: Aplikasi Streamlit Mode Loket (app/app.py)
```text
Buat app/app.py dengan Streamlit + streamlit-webrtc. Tata letak dua kolom. Kolom "Pengunjung": video webcam (mirror) dengan overlay landmark, huruf prediksi + keyakinan, kolom teks hasil isyarat, tombol Spasi/Hapus/Reset/Salin. Kolom "Petugas": kotak input untuk mengetik balasan, tombol "Tampilkan sebagai isyarat" (integrasi dengan text_to_sign.py jika sudah ada; jika belum, beri placeholder yang jelas). Tampilkan keterangan privasi dan disclaimer "alat bantu komunikasi dasar, bukan pengganti juru bahasa isyarat". Tangani error ramah jika model belum ada atau kamera ditolak browser. Jelaskan cara menjalankan dan masalah umum streamlit-webrtc (izin kamera, HTTPS/localhost).
```

### T7: Debugging (isi bagian dalam kurung)
```text
Aku mendapat error saat menjalankan perintah ini: [PERINTAH]
Pesan error lengkap: [TEMPEL ERROR LENGKAP]
File terkait: [TEMPEL ISI FILE]
Sistem operasi dan versi Python: [ISI]
Tolong: (1) jelaskan penyebabnya dengan bahasa sederhana, (2) berikan perbaikan minimal (jangan menulis ulang seluruh proyek), (3) jelaskan cara memastikan perbaikannya berhasil. Jika kamu butuh informasi tambahan untuk menebak penyebabnya, sebutkan perintah diagnostik yang harus kujalankan.
```

### T8: Dokumentasi dan bahan proposal
```text
Berdasarkan hasil proyek sampai sekarang, bantu aku menyusun bagian "Hasil dan Evaluasi" untuk proposal lomba. Berikut data nyata kami: [TEMPEL AKURASI, CONFUSION MATRIX/HURUF YANG SERING TERTUKAR, JUMLAH DATA, JUMLAH PENUTUR, HASIL UJI PENGGUNA JIKA ADA]. Tulis dengan jujur (termasuk kelemahan dan keterbatasan), hubungkan dengan SDG 4, 8, 9 dan Asta Cita 4, dan jangan menambahkan angka yang tidak kuberikan. Sertakan juga 5 pertanyaan sulit yang mungkin ditanyakan juri beserta saran jawabannya.
```

---

## Catatan untuk tim

- **Jangan percaya buta pada kode hasil AI.** Jalankan, baca, dan pahami setiap file. Kalau ada bagian yang tidak dimengerti, tanyakan ke Claude: "jelaskan baris demi baris bagian ini".
- **Commit ke Git setelah tiap paket tugas berhasil**, supaya mudah kembali jika ada yang rusak.
- **Satu orang menyimpan "sumber kebenaran"**: file `features.py` dan `config.py` hanya diubah lewat kesepakatan tim, karena semua modul bergantung padanya.
- **Verifikasi isyarat dengan penutur BISINDO asli.** Claude bisa salah soal bentuk huruf, jadi jangan jadikan Claude referensi isyarat.
