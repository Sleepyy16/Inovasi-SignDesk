# Bab Hasil dan Evaluasi — Proposal Inovasi Software SignDesk

> **Catatan Penggunaan:** Seluruh angka di bawah ini diambil murni dari hasil pengujian nyata repositori SignDesk (`train.py` & `evaluate.py`), tanpa angka yang dikarang. Jika Anda melakukan uji coba langsung bersama penutur Tuli/pengguna loket, tambahkan hasilnya pada subbab **Evaluasi Pengguna**.

---

## 1. Deskripsi Dataset dan Metodologi Evaluasi

Pengembangan prototipe **SignDesk (Fase MVP: Alfabet BISINDO A–Z)** menggunakan pendekatan ekstraksi titik kunci tangan (*hand landmarks*) melalui **MediaPipe Hands** (`max_num_hands=2`). Setiap sampel direpresentasikan sebagai vektor fitur berukuran tetap **127 angka** yang terdiri atas:
- Koordinat 3D `(x, y, z)` dari 21 titik per tangan untuk maksimal 2 tangan (126 angka) yang telah **diurutkan dari kiri ke kanan layar** berdasarkan posisi horizontal pergelangan tangan, kemudian **dinormalisasi secara bersama (*joint normalization*)** terhadap titik pusat dan rentang maksimum kedua tangan agar hubungan spasial antar-tangan pada huruf 2-tangan BISINDO tetap terjaga.
- Penanda jumlah tangan terdeteksi (`jumlah_tangan`: `1.0` atau `2.0`) sebagai fitur ke-127.

### Komposisi Data Nyata
Total data bersih yang digunakan dalam eksperimen adalah **2.596 sampel** yang mencakup seluruh **26 kelas huruf (A–Z)**:
1. **Dataset Citra Publik (`landmarks_dari_gambar.csv`)**: **1.138 sampel** berhasil diekstraksi dari 1.248 citra (110 citra dilewati karena tangan tidak terdeteksi oleh MediaPipe), yang berasal dari dua sumber:
   - Dataset Kaggle BISINDO: **281 sampel**
   - Dataset Mendeley BISINDO: **857 sampel**
   - Huruf dengan jumlah sampel citra paling sedikit adalah huruf **Z** (**31 sampel**).
2. **Dataset Rekaman Mandiri via Webcam (`landmarks_webcam.csv`)**: **1.458 sampel** dari **1 penutur mandiri (`person_id = P01`)**, dengan rata-rata **55–58 sampel per huruf**. Demi menjaga privasi, sistem hanya menyimpan vektor angka landmark beserta label huruf dan kode penutur tanpa menyimpan foto atau rekaman wajah.

### Skema Pengujian Lintas-Penutur (*Group Split*)
Untuk menghindari kebocoran data (*data leakage*) dan mengukur kemampuan generalisasi model pada pengguna baru di dunia nyata, pembagian data latih dan data uji dilakukan menggunakan **`GroupShuffleSplit` berdasarkan `person_id`** (bukan pembagian acak per baris):
- **Data Latih (`X_latih`)**: Seluruh data dari grup `kaggle` dan `mendeley` (**1.138 sampel** asli, yang diperkaya dengan **augmentasi pencerminan horizontal / *horizontal flip*** menjadi **2.276 sampel latih** untuk mengatasi perbedaan kamera *mirror* dan variasi tangan dominan kiri/kanan).
- **Data Uji (`X_uji`)**: Seluruh data webcam dari penutur **`P01`** (**1.458 sampel**). Dengan skema ini, model sama sekali belum pernah melihat satu pun frame dari kamera maupun tangan penutur `P01` selama pelatihan.

---

## 2. Hasil Eksperimen dan Perbandingan Model

Tabel berikut merangkum hasil evaluasi pada data uji lintas-penutur (`P01`, 1.458 sampel):

| Tahap Eksperimen | Model | Jumlah Sampel Latih | Jumlah Sampel Uji (`P01`) | Akurasi Uji | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sebelum augmentasi cermin | Random Forest (300 pohon) | 1.138 | 1.458 | 37,7% | 0,30 | 0,38 | 0,32 |
| Sebelum augmentasi cermin | MLP (128, 64) + StandardScaler | 1.138 | 1.458 | 35,0% | - | - | - |
| **Setelah augmentasi cermin** | **MLP (128, 64) + StandardScaler** | **2.276** | **1.458** | **88,4%** | **0,88** | **0,88** | **0,86** |

### Temuan Utama
1. **Dampak Krusial Augmentasi Cermin Horizontal pada BISINDO 2-Tangan**:
   Sebelum augmentasi cermin, akurasi lintas-penutur hanya mencapai **37,7%** karena banyak huruf BISINDO 2-tangan yang tidak simetris (seperti `B, D, G, K, L, M, N, P, Q, S, T, Y`) memiliki orientasi tangan kiri-kanan yang terbalik antara foto dataset publik (diambil dari kamera depan/belakang oleh fotografer) dan rekaman webcam pengguna. Setelah menambahkan pencerminan fitur horizontal (`cerminkan_fitur`), akurasi uji pada penutur `P01` meningkat drastis menjadi **88,4%** dengan model **MLP (`MLPClassifier`)**, melampaui target awal MVP sebesar 75%.
2. **Performa per Huruf**:
   - **23 dari 26 huruf (88,5% alfabet)** mencapai *recall* **≥ 86%** (di atas ambang target 70% per huruf).
   - **19 huruf mencapai *recall* sempurna 100%** pada penutur uji `P01`, yaitu: **B, C, E, F, G, H, I, J, M, N, P, Q, S, V, W, X**, serta 4 huruf lainnya berada di rentang 86%–98% (**A**: 86%, **K**: 93%, **L**: 93%, **O**: 98%, **R**: 95%, **T**: 98%, **Y**: 95%).

---

## 3. Analisis Kesalahan (*Error Analysis*), Kelemahan, dan Keterbatasan

Berdasarkan *Confusion Matrix* (`reports/confusion_matrix.png`), terdapat **5 pasangan huruf yang paling sering tertukar** pada pengujian lintas-domain murni (model dilatih dari foto publik, diuji pada webcam `P01`):

1. **`U` ditebak sebagai `H` (56 kali — *recall* `U` = 0%)**:
   - *Penyebab*: Pada dataset citra publik (`kaggle` dan `mendeley`), terdapat beberapa foto huruf `H` (isyarat 2 tangan) yang tangan keduanya tertutup atau gagal terdeteksi oleh MediaPipe sehingga tercatat sebagai isyarat **1 tangan** (`jumlah_tangan = 1.0`) dengan jari telunjuk tegak lurus ke atas. Saat penutur `P01` memperagakan huruf `U` (1 tangan) dengan telunjuk tegak, koordinatnya berimpit dengan sampel `H` yang gagal deteksi tangan kedua tersebut.
   - *Mitigasi*: Menambahkan penyaring konsistensi jumlah tangan (`jumlah_tangan`) pada tahap pemuatan data latih serta memperjelas lengkungan jari telunjuk-jempol pada huruf `U`.
2. **`D` ditebak sebagai `S` (32 kali) dan `P` (18 kali — *recall* `D` = 11%)**:
   - *Penyebab*: Huruf `D`, `S`, dan `P` pada BISINDO sama-sama menggunakan 2 tangan dengan jari yang membentuk lengkungan/kontak pada telunjuk tangan pasangan. Perbedaan sudut kemiringan kamera webcam terhadap dada/wajah menggeser proporsi koordinat vertikal (`y`) antar kedua tangan.
3. **`Z` ditebak sebagai `O` (28 kali) dan `E` (11 kali — *recall* `Z` = 30%)**:
   - *Penyebab*: Huruf `Z` memiliki jumlah data latih paling sedikit di dataset citra publik (hanya **31 sampel**) dan pada BISINDO melibatkan unsur sudut/gerakan yang pada fase MVP masih didekati sebagai pose statis.

### Keterbatasan Sistem Saat Ini (Jujur & Transparan)
- **Cakupan terbatas pada ejaan jari (*fingerspelling* A–Z)**: Sistem baru menerjemahkan alfabet untuk mengeja nama, kode, atau kata pendek di loket layanan, belum mencakup kosakata/kalimat utuh BISINDO maupun ekspresi non-manual.
- **Huruf dinamis (`J` dan `Z`) masih bersifat eksperimental**: Diperlakukan sebagai pose akhir statis dan telah diberi penanda `[eksperimental]` secara eksplisit pada antarmuka pengguna.
- **Jumlah penutur webcam masih terbatas (`1 penutur mandiri / P01`)**: Diperlukan penambahan variasi penutur webcam (`P02`, `P03`, dst.) dan verifikasi langsung bersama penutur Tuli pengguna BISINDO.
- **Ketergantungan pada pencahayaan dan *occlusion***: Ketika dua tangan saling menutupi (*self-occlusion*) secara ekstrem di depan kamera atau latar belakang pakaian sangat mirip dengan warna kulit, MediaPipe Hands dapat kehilangan deteksi salah satu tangan.

---

## 4. Keterkaitan dengan SDGs dan Asta Cita

1. **SDG 4 (Pendidikan Berkualitas)**:
   Menyediakan sarana literasi dua arah bagi petugas layanan pendidikan/kampus dan mahasiswa/pelajar Tuli, sekaligus memperkenalkan bentuk alfabet BISINDO secara visual melalui fitur *Teks ke Isyarat* pada Mode Loket.
2. **SDG 8 (Pekerjaan Layak dan Pertumbuhan Ekonomi)**:
   Menurunkan hambatan komunikasi di loket pelayanan kerja, perbankan, dan administrasi publik sehingga penyandang disabilitas rungu/wicara memiliki akses yang lebih setara terhadap layanan ketenagakerjaan dan ekonomi.
3. **SDG 9 (Industri, Inovasi, dan Infrastruktur)**:
   Membangun infrastruktur layanan publik digital yang inklusif namun **ringan (*lightweight*)**: berjalan di CPU laptop standar loket menggunakan ekstraksi fitur geometris 127 dimensi + `MLP` tanpa memerlukan GPU mahal atau koneksi server eksternal, serta menjamin **privasi penuh (*on-device processing*)** karena video tidak pernah disimpan.
4. **Asta Cita 4 (Memperkuat pembangunan sumber daya manusia, sains, teknologi, pendidikan, kesehatan, serta penguatan peran perempuan, pemuda, dan penyandang disabilitas)**:
   Mendukung aksesibilitas layanan publik (kesehatan, administrasi kependudukan, pendidikan) bagi penyandang disabilitas Tuli di Indonesia dengan mengutamakan **BISINDO** sebagai bahasa isyarat yang tumbuh di komunitas Tuli Indonesia, dilengkapi etika penggunaan yang jelas bahwa sistem ini adalah *"alat bantu komunikasi dasar, bukan pengganti juru bahasa isyarat."*

---

## 5. Lima Pertanyaan Sulit Juri dan Saran Jawaban

### Pertanyaan 1: *"Mengapa kalian hanya menerjemahkan alfabet (A–Z), padahal percakapan sehari-hari penutur Tuli menggunakan kosakata isyarat dan ekspresi wajah?"*
- **Saran Jawaban**:
  "Betul sekali, Bapak/Ibu. Dalam skenario **loket layanan publik** (seperti pendaftaran pasien, pembuatan KTP, atau loket akademik), informasi paling krusial yang sering salah ditangkap oleh petugas yang tidak bisa berbahasa isyarat adalah **data spesifik yang harus dieja**, seperti **nama lengkap, inisial, kode booking, atau nama jalan**. Oleh karena itu, pada fase MVP ini kami memfokuskan SignDesk pada *fingerspelling* alfabet A–Z BISINDO yang akurat untuk 1 maupun 2 tangan, sambil mencantumkan disclaimer tegas di aplikasi bahwa SignDesk adalah *alat bantu komunikasi dasar di loket, bukan pengganti Juru Bahasa Isyarat (JBI)*."

### Pertanyaan 2: *"Banyak proyek AI bahasa isyarat mengklaim akurasi 99%, mengapa akurasi kalian 88,4% dan masih ada huruf seperti D dan Z yang rendah?"*
- **Saran Jawaban**:
  "Angka 99% pada banyak penelitian pemula biasanya diperoleh dari **pembagian data acak (*random split*)**, di mana frame ke-1 dan frame ke-2 dari video orang yang sama masuk ke data latih dan data uji sekaligus — itu adalah *data leakage*. Di SignDesk, kami menggunakan evaluasi yang jujur dengan **`GroupShuffleSplit` berdasarkan `person_id`**: model dilatih 100% dari dataset foto publik (Kaggle & Mendeley, 1.138 sampel) dan diuji pada 1.458 sampel webcam dari penutur baru (`P01`) yang belum pernah dilihat model sama sekali. Dalam pengujian lintas-domain murni ini, 23 dari 26 huruf sudah mencapai recall 86%–100% dengan rata-rata **88,4%**. Kami juga melaporkan secara transparan 3 huruf yang masih lemah (`U`, `D`, `Z`) beserta akar masalah teknisnya pada *confusion matrix*."

### Pertanyaan 3: *"Kenapa tidak memakai Deep Learning berbasis CNN/Vision Transformer langsung pada gambar kamera, melainkan memakai MediaPipe + MLP?"*
- **Saran Jawaban**:
  "Ada tiga alasan rekayasa (*engineering*) untuk kebutuhan loket layanan publik:
  1. **Kecepatan di perangkat standar**: Laptop loket kelurahan/puskesmas umumnya tidak memiliki GPU diskrit. Ekstraksi 127 angka landmark + `MLPClassifier` berjalan *real-time* di CPU biasa dengan latensi milidetik.
  2. **Ketahanan terhadap latar belakang**: Model CNN gambar mentah mudah mengalami *overfitting* pada warna baju atau tembok di belakang pengunjung. Dengan normalisasi bersama (*joint normalization*) pada titik landmark, model hanya fokus pada geometri relatif kedua tangan.
  3. **Privasi**: Saat mengumpulkan data latih baru (`collect_data.py`), kami hanya menyimpan 127 angka koordinat ke CSV tanpa pernah menyimpan foto wajah atau tubuh pengunjung."

### Pertanyaan 4: *"Bagaimana sistem kalian menangani perbedaan tangan kiri/kanan (kidal) dan perbedaan huruf 1 tangan vs 2 tangan di BISINDO?"*
- **Saran Jawaban**:
  "Kami merancang tiga mekanisme khusus di `src/features.py` dan `src/train.py`:
  1. Kami **tidak bergantung** pada label `Left`/`Right` bawaan MediaPipe yang sering terbalik saat kamera dicerminkan, melainkan mengurutkan tangan berdasarkan koordinat horizontal `x` pergelangan tangan di layar.
  2. Kami melakukan **normalisasi bersama (*joint normalization*)** untuk kedua tangan sekaligus sehingga jarak dan posisi relatif tangan kiri terhadap tangan kanan tidak hilang.
  3. Kami menambahkan **augmentasi pencerminan horizontal (`cerminkan_fitur`)** pada data latih yang membalik sumbu `x` sekaligus menukar urutan slot tangan 1 dan tangan 2, yang terbukti menaikkan akurasi lintas-penutur kami dari **37,7% menjadi 88,4%**."

### Pertanyaan 5: *"Bagaimana mencegah huruf masuk berulang-ulang atau salah ketik saat tangan pengunjung baru bergerak berganti huruf di depan kamera?"*
- **Saran Jawaban**:
  "Kami membangun modul penstabil prediksi (`src/stabilizer.py`) dengan 4 lapis pengaman:
  1. **Ambang keyakinan (*confidence threshold* 70%)** untuk membuang prediksi saat tangan masih bergerak/transisi.
  2. **Voting mayoritas 10 frame terakhir** sehingga 1–2 frame *noise* tidak mengubah huruf.
  3. **Konfirmasi waktu stabil 1,0 detik** (dilengkapi indikator *progress bar* visual di layar agar pengunjung tahu kapan hurufnya terkunci).
  4. **Jeda huruf sama (*cooldown* 1,0 detik)** serta tombol koreksi (**Spasi, Hapus, Reset**) agar menahan tangan terlalu lama tidak menghasilkan huruf ganda seperti `AAAA`."
