# ============================================================
# SignDesk - config.py
# SATU tempat untuk SEMUA pengaturan proyek.
# Ubah nilai di sini, efeknya langsung terasa di seluruh kode.
#
# Aturan tim: file ini hanya diubah lewat kesepakatan bersama.
# Jangan hardcode nilai-nilai ini di file lain!
# ============================================================

import os

# ------------------------------------------------------------------
# HURUF YANG DIDUKUNG
# ------------------------------------------------------------------
# Daftar 26 huruf alfabet BISINDO A-Z.
# J dan Z ditandai eksperimental karena melibatkan gerakan tangan.
HURUF_DIDUKUNG = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

HURUF_EKSPERIMENTAL = ["J", "Z"]  # Belum akurat karena perlu data gerakan

# ------------------------------------------------------------------
# PATH FOLDER
# (semua relatif terhadap root proyek: folder signdesk/)
# ------------------------------------------------------------------

# Root proyek: folder tempat config.py ini berada
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder data
DATA_DIR        = os.path.join(ROOT_DIR, "data")
RAW_IMAGES_DIR  = os.path.join(DATA_DIR, "raw_images")  # gambar dataset
LANDMARKS_DIR   = os.path.join(DATA_DIR, "landmarks")   # CSV hasil ekstraksi

# Folder model terlatih
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Folder gambar isyarat untuk Teks → Isyarat
ASSETS_ALFABET_DIR = os.path.join(ROOT_DIR, "assets", "alfabet")

# Folder laporan evaluasi
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# Nama file CSV gabungan landmark
LANDMARKS_CSV = os.path.join(LANDMARKS_DIR, "landmarks_gabungan.csv")

# Nama file CSV hasil pengumpulan data webcam (src/collect_data.py)
LANDMARKS_WEBCAM_CSV = os.path.join(LANDMARKS_DIR, "landmarks_kamera.csv")

# Interval minimum antar rekaman sampel otomatis saat tombol ditahan (detik)
INTERVAL_REKAM_DETIK = 0.15

# Nama file model terlatih (classifier tangan)
MODEL_PATH       = os.path.join(MODELS_DIR, "model_terbaik.joblib")
MODEL_LABEL_PATH = os.path.join(MODELS_DIR, "daftar_label.joblib")

# Model MediaPipe HandLandmarker (file .task, diunduh dari Google)
# Unduh dari: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
HAND_LANDMARKER_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")

# ------------------------------------------------------------------
# PENGATURAN KAMERA / MEDIAPIPE
# ------------------------------------------------------------------

# Indeks kamera yang dipakai (0 = kamera utama/pertama)
# Ganti ke 1 atau 2 jika kamera yang benar bukan indeks 0
INDEKS_KAMERA = 0

# Resolusi kamera (lebar x tinggi) dalam piksel
LEBAR_KAMERA  = 640
TINGGI_KAMERA = 480

# Jumlah tangan maksimum yang dideteksi MediaPipe
MAX_TANGAN = 2

# Tingkat kepercayaan deteksi dan tracking MediaPipe (0.0 - 1.0)
KEYAKINAN_DETEKSI  = 0.7
KEYAKINAN_TRACKING = 0.5

# ------------------------------------------------------------------
# PENGATURAN EKSTRAKSI FITUR
# ------------------------------------------------------------------

# Jumlah titik landmark per tangan (standar MediaPipe = 21)
TITIK_PER_TANGAN = 21

# Komponen koordinat per titik (x, y, z)
KOMPONEN_PER_TITIK = 3

# Ukuran fitur untuk SATU tangan (21 titik × 3 koordinat = 63)
FITUR_PER_TANGAN = TITIK_PER_TANGAN * KOMPONEN_PER_TITIK  # = 63

# Ukuran fitur untuk DUA tangan (63 × 2 = 126)
FITUR_DUA_TANGAN = FITUR_PER_TANGAN * MAX_TANGAN  # = 126

# +1 untuk fitur penanda jumlah tangan (1.0 = satu tangan, 2.0 = dua tangan)
UKURAN_VEKTOR_FITUR = FITUR_DUA_TANGAN + 1  # = 127

# ------------------------------------------------------------------
# PENGATURAN PENSTABIL PREDIKSI (src/stabilizer.py)
# ------------------------------------------------------------------

# Jumlah frame terakhir yang disimpan untuk voting mayoritas
N_FRAME_VOTING = 10

# Keyakinan minimum agar prediksi dianggap valid (0.0 - 1.0)
AMBANG_KEYAKINAN = 0.70

# Waktu (detik) prediksi harus stabil sebelum huruf dikonfirmasi
WAKTU_STABIL_DETIK = 1.0

# Jeda minimum (detik) setelah huruf dikonfirmasi sebelum bisa konfirmasi lagi
# (mencegah huruf yang sama terus-menerus masuk)
JEDA_SETELAH_KONFIRMASI_DETIK = 0.5

# ------------------------------------------------------------------
# PENGATURAN PELATIHAN MODEL (src/train.py)
# ------------------------------------------------------------------

# Proporsi data yang dipakai untuk pengujian (0.0 - 1.0)
PROPORSI_UJI = 0.2

# Random seed agar hasil pelatihan bisa direproduksi
RANDOM_SEED = 42

# Jumlah pohon untuk RandomForest
RF_N_ESTIMATORS = 200

# Ukuran layer tersembunyi untuk MLP
MLP_HIDDEN_LAYER_SIZES = (256, 128)

# Jumlah iterasi maksimum MLP
MLP_MAX_ITER = 500

# ------------------------------------------------------------------
# PENGATURAN APLIKASI STREAMLIT (app/app.py)
# ------------------------------------------------------------------

# Interval refresh antarmuka Streamlit (detik)
INTERVAL_REFRESH = 0.1

# Teks disclaimer yang wajib ditampilkan di UI
TEKS_DISCLAIMER = (
    "⚠️ Alat bantu komunikasi dasar — bukan pengganti juru bahasa isyarat. "
    "Video diproses di perangkat dan tidak disimpan."
)

