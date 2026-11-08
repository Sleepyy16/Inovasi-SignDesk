"""Semua pengaturan SignDesk ada di sini.

Kenapa satu file? Supaya training, demo real-time, dan aplikasi memakai angka
yang sama. Ubah angka di sini, jangan di-hardcode di file lain.
Catatan tim: file ini hanya diubah lewat kesepakatan bersama.
"""

from pathlib import Path

# ---------- Path folder ----------
# Path dihitung dari lokasi file ini, jadi aman dijalankan dari folder mana pun.
BASE_DIR = Path(__file__).resolve().parent
RAW_IMAGES_DIR = BASE_DIR / "data" / "raw_images"
LANDMARKS_DIR = BASE_DIR / "data" / "landmarks"
MODELS_DIR = BASE_DIR / "models"
ASSETS_ALFABET_DIR = BASE_DIR / "assets" / "alfabet"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODELS_DIR / "model_terbaik.joblib"

# ---------- Label ----------
HURUF = [chr(kode) for kode in range(ord("A"), ord("Z") + 1)]  # A-Z
HURUF_EKSPERIMENTAL = ["J", "Z"]  # kemungkinan bergerak; di MVP dianggap pose statis

# ---------- Kamera ----------
INDEKS_KAMERA = 0        # coba 1 atau 2 jika kamera tidak terbuka
LEBAR_KAMERA = 640
TINGGI_KAMERA = 480
CERMINKAN_KAMERA = True  # tampilan mirror agar natural bagi pengguna

# ---------- MediaPipe ----------
MAKS_TANGAN = 2
MIN_KEYAKINAN_DETEKSI = 0.5
MIN_KEYAKINAN_TRACKING = 0.5

# ---------- Penstabil prediksi ----------
JUMLAH_FRAME_VOTING = 10      # N prediksi terakhir untuk voting
AMBANG_KEYAKINAN = 0.7        # di bawah ini prediksi diabaikan
DETIK_STABIL = 1.0            # huruf harus stabil selama ini sebelum masuk teks
DETIK_JEDA_HURUF_SAMA = 1.0   # jeda/perubahan sebelum huruf sama boleh masuk lagi

# ---------- Ekstraksi dari dataset gambar ----------
EKSTENSI_GAMBAR = (".jpg", ".jpeg", ".png", ".bmp")
FILE_CSV_GAMBAR = LANDMARKS_DIR / "landmarks_dari_gambar.csv"
# Regex untuk mengambil person_id dari NAMA FILE (grup pertama = id).
# Contoh: r"^(P\d+)_" untuk file "P01_A_003.jpg". None = tidak dipakai (person_id = "unknown").
POLA_PERSON_ID = None

# ---------- Pengumpulan data webcam ----------
FILE_CSV_WEBCAM = LANDMARKS_DIR / "landmarks_webcam.csv"
JEDA_REKAM_DETIK = 0.15            # jarak minimal antar sampel saat SPASI ditahan
MAKS_SAMPEL_PER_HURUF_SESI = 150   # batas wajar per huruf per sesi (agar data bervariasi)

# Apakah fitur dihitung dari frame yang sudah DICERMINKAN (seperti yang dilihat pengguna)?
# WAJIB sama di pengumpulan data dan real-time. Dataset gambar tidak dicerminkan,
# jadi default False (deteksi pada frame asli, mirror hanya untuk tampilan).
# Jika akurasi real-time buruk padahal evaluasi bagus, coba ubah ke True lalu uji ulang.
FITUR_DARI_FRAME_CERMIN = False

# ---------- Pelatihan ----------
RASIO_UJI = 0.2
RANDOM_SEED = 42
RF_JUMLAH_POHON = 300
MLP_LAPISAN = (128, 64)
MLP_MAKS_ITERASI = 500
FILE_GAMBAR_CONFUSION = REPORTS_DIR / "confusion_matrix.png"
