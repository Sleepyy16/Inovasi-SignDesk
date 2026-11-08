"""
SignDesk - src/features.py
==========================
Modul TERPUSAT untuk ekstraksi fitur landmark tangan.

PENTING: File ini dipakai oleh SEMUA modul lain:
  - src/extract_from_images.py  (dari dataset gambar)
  - src/collect_data.py         (dari webcam live)
  - src/realtime.py             (prediksi real-time)
  - app/app.py                  (aplikasi Streamlit)

Jangan ubah format output fungsi `ekstrak_fitur()` tanpa
koordinasi dengan semua modul di atas, karena perubahan kecil
bisa menyebabkan "model akurat saat training, ngawur saat dipakai".

CATATAN VERSI:
  Modul ini menggunakan MediaPipe Tasks API (versi 1.x ke atas).
  API ini memerlukan file model hand_landmarker.task yang sudah
  tersimpan di folder models/.
  (MediaPipe 0.x memakai mp.solutions.hands yang berbeda.)

ALUR EKSTRAKSI (5 langkah):
  Frame BGR
    │ (1) Konversi ke mp.Image (format MediaPipe)
    ▼
  Landmark mentah (21 titik per tangan, x/y/z relatif gambar)
    │ (2) Urutkan tangan berdasarkan posisi x pergelangan (kiri → kanan)
    ▼
  Titik terurut
    │ (3) Normalisasi bersama: geser ke pusat, skala ke [-1, +1]
    ▼
  Titik ternormalisasi
    │ (4) Padding: jika hanya 1 tangan, slot tangan kedua diisi nol
    │ (5) Tambah fitur jumlah tangan (1.0 atau 2.0)
    ▼
  Vektor fitur 127 angka [float32]
"""

import sys
import os
import time

import numpy as np

# Tambahkan root proyek ke path agar bisa import config
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import config

# ──────────────────────────────────────────────────────────────────
# Import MediaPipe Tasks API (versi 1.x)
# ──────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision.core import (
        vision_task_running_mode as _running_mode_module,
    )
    _RunningMode    = _running_mode_module.VisionTaskRunningMode
    _HandLandmarker = mp_vision.HandLandmarker
    _HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
    _BaseOptions    = mp_python.BaseOptions
    _mp_Image       = mp.Image
    _mp_ImageFormat = mp.ImageFormat
except ImportError as e:
    raise ImportError(
        f"Gagal mengimport MediaPipe: {e}\n"
        "Jalankan: pip install mediapipe"
    ) from e

# Koneksi antar landmark tangan (untuk menggambar garis skeleton)
# Pasangan indeks titik yang terhubung (dari dokumentasi MediaPipe)
_KONEKSI_TANGAN = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # ibu jari
    (0, 5), (5, 6), (6, 7), (7, 8),          # telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),      # tengah
    (9, 13), (13, 14), (14, 15), (15, 16),    # manis
    (13, 17), (17, 18), (18, 19), (19, 20),   # kelingking
    (0, 17),                                   # telapak bawah
]


# ──────────────────────────────────────────────────────────────────
# FUNGSI PUBLIK 1: buat_detektor()
# ──────────────────────────────────────────────────────────────────

def buat_detektor(mode_gambar_statis: bool = False) -> _HandLandmarker:
    """
    Membuat dan mengembalikan objek MediaPipe HandLandmarker.

    Parameters
    ----------
    mode_gambar_statis : bool
        True  → untuk memproses gambar satu per satu (dataset, offline).
                 Gunakan RunningMode.IMAGE. Tiap gambar dideteksi ulang.
        False → untuk video real-time (webcam). RunningMode.VIDEO.
                 Memanfaatkan tracking antar frame untuk kecepatan.

    Returns
    -------
    mediapipe HandLandmarker
        Objek detektor yang siap dipakai. Gunakan sebagai context manager:

        with buat_detektor(mode_gambar_statis=True) as detektor:
            fitur, hasil = ekstrak_fitur(frame, detektor)

    Prasyarat
    ---------
    File models/hand_landmarker.task harus ada.
    Unduh dengan perintah:
        curl -L -o models/hand_landmarker.task \\
          https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    """
    # Cek apakah file model ada
    if not os.path.isfile(config.HAND_LANDMARKER_MODEL):
        raise FileNotFoundError(
            f"File model MediaPipe tidak ditemukan:\n"
            f"  {config.HAND_LANDMARKER_MODEL}\n\n"
            f"Unduh dengan perintah:\n"
            f"  curl -L -o models/hand_landmarker.task \\\n"
            f"    https://storage.googleapis.com/mediapipe-models/"
            f"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    # Pilih mode berdasarkan kebutuhan
    if mode_gambar_statis:
        running_mode = _RunningMode.IMAGE
    else:
        # VIDEO mode: butuh timestamp saat memanggil detect_for_video()
        running_mode = _RunningMode.VIDEO

    opsi = _HandLandmarkerOptions(
        base_options             = _BaseOptions(
                                       model_asset_path=config.HAND_LANDMARKER_MODEL
                                   ),
        running_mode             = running_mode,
        num_hands                = config.MAX_TANGAN,
        min_hand_detection_confidence = config.KEYAKINAN_DETEKSI,
        min_hand_presence_confidence  = config.KEYAKINAN_DETEKSI,
        min_tracking_confidence       = config.KEYAKINAN_TRACKING,
    )

    detektor = _HandLandmarker.create_from_options(opsi)
    # Simpan penanda mode agar ekstrak_fitur() memanggil metode yang tepat
    detektor._mode_gambar_statis = mode_gambar_statis
    return detektor


# ──────────────────────────────────────────────────────────────────
# FUNGSI PUBLIK 2: ekstrak_fitur()   ← INI FUNGSI UTAMA
# ──────────────────────────────────────────────────────────────────

def ekstrak_fitur(
    frame_bgr:  np.ndarray,
    detektor:   _HandLandmarker,
    timestamp_ms: int | None = None,
) -> tuple:
    """
    Mengekstrak vektor fitur dari satu frame kamera atau gambar.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Frame dalam format BGR (seperti yang dibaca OpenCV).
        Shape: (tinggi, lebar, 3).

    detektor : HandLandmarker
        Objek detektor yang dibuat dengan buat_detektor().
        Gunakan SATU objek untuk banyak frame (lebih efisien).

    timestamp_ms : int atau None
        Timestamp dalam milidetik.
        - Untuk mode_gambar_statis=True: boleh None.
        - Untuk mode video: disarankan diisi, harus selalu meningkat tiap frame.
          Gunakan: int(time.time() * 1000)

    Returns
    -------
    tuple (vektor_fitur, hasil_mediapipe)
        vektor_fitur    : np.ndarray shape (127,) dtype float32,
                          atau None jika tidak ada tangan terdeteksi.
        hasil_mediapipe : objek HandLandmarkerResult (berguna untuk overlay),
                          atau None jika gagal.
    """
    # Konversi frame BGR → format mp.Image (RGB)
    mp_image = _bgr_ke_mp_image(frame_bgr)

    # Jalankan deteksi sesuai mode yang ditentukan saat pembuatan detektor
    mode_statis = getattr(detektor, "_mode_gambar_statis", None)

    if mode_statis is True:
        hasil = detektor.detect(mp_image)
    elif mode_statis is False:
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        hasil = detektor.detect_for_video(mp_image, timestamp_ms)
    else:
        # Fallback jika detektor dibuat tanpa helper buat_detektor()
        try:
            if timestamp_ms is not None:
                hasil = detektor.detect_for_video(mp_image, timestamp_ms)
            else:
                hasil = detektor.detect(mp_image)
        except ValueError:
            if timestamp_ms is None:
                timestamp_ms = int(time.time() * 1000)
            try:
                hasil = detektor.detect_for_video(mp_image, timestamp_ms)
            except ValueError:
                hasil = detektor.detect(mp_image)

    # Jika tidak ada tangan terdeteksi
    if not hasil.hand_landmarks:
        return None, hasil

    # Kumpulkan landmark dari semua tangan
    semua_tangan = _kumpulkan_landmark(hasil)

    # Urutkan tangan berdasarkan posisi x pergelangan (kiri → kanan)
    semua_tangan = _urutkan_tangan(semua_tangan)

    # Normalisasi bersama
    semua_tangan_norm = _normalisasi_bersama(semua_tangan)

    # Susun vektor fitur 127 angka
    vektor = _susun_vektor(semua_tangan_norm)

    return vektor, hasil


# ──────────────────────────────────────────────────────────────────
# FUNGSI PUBLIK 3: gambar_overlay()
# ──────────────────────────────────────────────────────────────────

def gambar_overlay(
    frame_bgr:      np.ndarray,
    hasil_mediapipe,
) -> np.ndarray:
    """
    Menggambar landmark dan koneksi tangan di atas frame BGR.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Frame asli BGR. Dimodifikasi langsung (in-place).
    hasil_mediapipe : HandLandmarkerResult
        Objek hasil dari ekstrak_fitur() (nilai kedua tuple).

    Returns
    -------
    np.ndarray
        Frame yang sama dengan overlay sudah digambar.
    """
    if hasil_mediapipe is None or not hasil_mediapipe.hand_landmarks:
        return frame_bgr

    try:
        import cv2
    except ImportError:
        return frame_bgr  # tidak bisa menggambar tanpa OpenCV

    h, w = frame_bgr.shape[:2]

    for landmarks_tangan in hasil_mediapipe.hand_landmarks:
        # Konversi koordinat relatif (0–1) ke piksel
        titik_piksel = [
            (int(lm.x * w), int(lm.y * h))
            for lm in landmarks_tangan
        ]

        # Gambar garis koneksi (skeleton)
        for (a, b) in _KONEKSI_TANGAN:
            if a < len(titik_piksel) and b < len(titik_piksel):
                cv2.line(frame_bgr, titik_piksel[a], titik_piksel[b],
                         (200, 200, 200), 1, cv2.LINE_AA)

        # Gambar titik landmark
        for px, py in titik_piksel:
            cv2.circle(frame_bgr, (px, py), 4, (0, 255, 0), -1, cv2.LINE_AA)

        # Tandai pergelangan (landmark 0) lebih besar
        if titik_piksel:
            cv2.circle(frame_bgr, titik_piksel[0], 7, (0, 128, 255), -1, cv2.LINE_AA)

    return frame_bgr


# ──────────────────────────────────────────────────────────────────
# FUNGSI INTERNAL (awalan _) — tidak perlu dipakai langsung di luar
# ──────────────────────────────────────────────────────────────────

def _bgr_ke_mp_image(frame_bgr: np.ndarray) -> _mp_Image:
    """
    Mengonversi frame OpenCV (BGR, uint8) ke objek mp.Image (RGB).

    MediaPipe Tasks API tidak menerima numpy array langsung —
    harus dibungkus dalam objek mp.Image dengan format RGB.

    Langkah:
      BGR numpy → flip channel → RGB numpy → mp.Image
    """
    # Balik urutan channel: BGR → RGB
    frame_rgb = frame_bgr[:, :, ::-1].copy()

    # Bungkus dalam mp.Image
    return _mp_Image(image_format=_mp_ImageFormat.SRGB, data=frame_rgb)


def _kumpulkan_landmark(hasil) -> list:
    """
    Mengubah format landmark MediaPipe Tasks API ke list numpy array.

    Dalam Tasks API, hasil.hand_landmarks adalah list of list:
      hasil.hand_landmarks[i] = list dari 21 objek NormalizedLandmark
      Setiap NormalizedLandmark punya .x, .y, .z (rentang 0.0–1.0)

    Returns
    -------
    list of np.ndarray shape (21, 3)
    """
    semua_tangan = []
    for landmarks_tangan in hasil.hand_landmarks:
        titik = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks_tangan],
            dtype=np.float32,
        )  # shape: (21, 3)
        semua_tangan.append(titik)
    return semua_tangan


def _urutkan_tangan(semua_tangan: list) -> list:
    """
    Mengurutkan tangan berdasarkan posisi x pergelangan (kiri → kanan).

    KENAPA tidak pakai label "Left"/"Right" dari MediaPipe?
    -------------------------------------------------------
    Label MediaPipe bisa tertukar saat gambar dicerminkan (mirror).
    Dengan mengurutkan berdasarkan posisi x, kita konsisten tanpa
    peduli label tersebut.

    Titik pergelangan = landmark indeks 0 (WRIST).
    x mendekati 0.0 = sisi kiri gambar, x mendekati 1.0 = sisi kanan.

    Contoh:
      Tangan A: pergelangan di x = 0.7  (kanan layar)
      Tangan B: pergelangan di x = 0.3  (kiri layar)
      Setelah diurutkan: [B, A]  (kiri dulu, kanan kemudian)
    """
    return sorted(semua_tangan, key=lambda tangan: tangan[0, 0])


def _normalisasi_bersama(semua_tangan: list) -> list:
    """
    Normalisasi posisi dan skala BERSAMA untuk semua tangan.

    KENAPA dinormalisasi bersama (bukan per tangan)?
    -------------------------------------------------
    Untuk huruf dua tangan BISINDO, posisi RELATIF antara kedua tangan
    itu penting. Kalau setiap tangan dinormalisasi sendiri-sendiri,
    informasi "seberapa jauh tangan kiri dari tangan kanan" akan hilang.

    LANGKAH-LANGKAH (dengan contoh angka sederhana):
    -------------------------------------------------

    Misal 2 tangan, masing-masing 2 titik:
      Tangan kiri:  A=(0.2, 0.3)  B=(0.4, 0.5)
      Tangan kanan: C=(0.6, 0.3)  D=(0.8, 0.7)
    Semua titik: [(0.2,0.3), (0.4,0.5), (0.6,0.3), (0.8,0.7)]

    Langkah 1 — Hitung pusat (rata-rata semua titik):
      pusat_x = (0.2+0.4+0.6+0.8)/4 = 0.5
      pusat_y = (0.3+0.5+0.3+0.7)/4 = 0.45

    Langkah 2 — Geser semua titik agar pusatnya di (0, 0):
      A' = (0.2-0.5, 0.3-0.45) = (-0.3, -0.15)
      D' = (0.8-0.5, 0.7-0.45) = (+0.3, +0.25)

    Langkah 3 — Hitung skala = jarak terbesar dari pusat (pakai x,y saja):
      jarak D' = sqrt(0.3² + 0.25²) ≈ 0.391  ← terbesar
      skala = 0.391

    Langkah 4 — Bagi semua titik dengan skala:
      A'' ≈ (-0.767, -0.384)
      D'' ≈ (+0.767, +0.639)
    Semua nilai x,y masuk rentang [-1, +1].
    """
    if not semua_tangan:
        return semua_tangan

    # Gabungkan semua titik menjadi satu array besar
    semua_titik = np.vstack(semua_tangan)  # shape: (N_tangan × 21, 3)

    # Langkah 1 & 2: Hitung pusat dan geser
    pusat = semua_titik.mean(axis=0)         # shape: (3,)
    semua_titik_geser = semua_titik - pusat  # shape: (N×21, 3)

    # Langkah 3: Hitung skala dari jarak x,y saja
    jarak = np.linalg.norm(semua_titik_geser[:, :2], axis=1)
    skala = jarak.max()

    # Jaga-jaga: hindari pembagian dengan nol
    if skala < 1e-6:
        skala = 1.0

    # Langkah 4: Bagi dengan skala
    semua_titik_norm = semua_titik_geser / skala

    # Kembalikan dalam format list per tangan
    hasil = []
    for i in range(len(semua_tangan)):
        mulai = i * config.TITIK_PER_TANGAN
        akhir = mulai + config.TITIK_PER_TANGAN
        hasil.append(semua_titik_norm[mulai:akhir])  # shape: (21, 3)

    return hasil


def _susun_vektor(semua_tangan_norm: list) -> np.ndarray:
    """
    Menyusun vektor fitur akhir berukuran 127.

    Tata letak (127 angka):
    ┌──────────────────────────────────────────────────────────┐
    │  [0  … 62]  → 63 angka fitur tangan pertama (kiri)      │
    │  [63 … 125] → 63 angka fitur tangan kedua  (kanan)      │
    │  [126]      → 1 angka: jumlah tangan (1.0 atau 2.0)     │
    └──────────────────────────────────────────────────────────┘

    Jika hanya 1 tangan:
    - Slot tangan yang ada diisi sesuai posisinya
    - Slot tangan yang kosong diisi NOL (padding)
    - Angka terakhir = 1.0

    KENAPA padding nol?
    -------------------
    Model ML butuh input berukuran tetap. Nol dipilih karena setelah
    normalisasi, nol berarti "di pusat/tidak ada". Model akan belajar
    bahwa kombinasi nol-nol-nol di slot tangan berarti tangan tidak ada.
    """
    jumlah_tangan = len(semua_tangan_norm)

    # Buat dua slot kosong (masing-masing 63 angka), diisi nol
    slot_tangan_1 = np.zeros(config.FITUR_PER_TANGAN, dtype=np.float32)
    slot_tangan_2 = np.zeros(config.FITUR_PER_TANGAN, dtype=np.float32)

    if jumlah_tangan >= 1:
        # .flatten() mengubah shape (21, 3) → (63,)
        slot_tangan_1 = semua_tangan_norm[0].flatten().astype(np.float32)

    if jumlah_tangan >= 2:
        slot_tangan_2 = semua_tangan_norm[1].flatten().astype(np.float32)

    # Fitur penanda jumlah tangan (0.0, 1.0, atau 2.0)
    fitur_jumlah = np.array([float(jumlah_tangan)], dtype=np.float32)

    # Gabungkan: [tangan_1 (63) | tangan_2 (63) | jumlah (1)] = 127
    vektor = np.concatenate([slot_tangan_1, slot_tangan_2, fitur_jumlah])

    # Validasi ukuran (bug catcher)
    assert vektor.shape == (config.UKURAN_VEKTOR_FITUR,), (
        f"BUG: ukuran vektor {vektor.shape} != {config.UKURAN_VEKTOR_FITUR}. "
        f"Periksa config.py."
    )

    return vektor


# ──────────────────────────────────────────────────────────────────
# NAMA KOLOM (untuk header CSV di extract_from_images.py)
# ──────────────────────────────────────────────────────────────────

def nama_kolom_fitur() -> list:
    """
    Mengembalikan daftar nama kolom fitur untuk header CSV.

    Format: f_000, f_001, ..., f_126  (127 nama)

    Pemakaian:
        kolom = ['label', 'person_id', 'source'] + nama_kolom_fitur()
        df.to_csv(path, header=kolom, index=False)
    """
    return [f"f_{i:03d}" for i in range(config.UKURAN_VEKTOR_FITUR)]
