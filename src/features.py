"""Ekstraksi fitur tangan (DIPAKAI BERSAMA oleh training dan real-time).

Kenapa satu file bersama? Bug paling umum di proyek seperti ini adalah
"akurat saat training, ngawur saat dipakai" karena cara menghitung fitur
berbeda. Dengan satu fungsi yang sama, hal itu tidak bisa terjadi.

Alur:
  gambar -> MediaPipe Hands -> landmark tiap tangan (21 titik x,y,z)
         -> urutkan tangan kiri ke kanan di layar
         -> normalisasi bersama (pusat + skala)
         -> padding nol jika 1 tangan + penanda jumlah tangan
         -> vektor berukuran tetap (127 angka)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

import config

TITIK_PER_TANGAN = 21
NILAI_PER_TITIK = 3  # x, y, z
UKURAN_TANGAN = TITIK_PER_TANGAN * NILAI_PER_TITIK  # 63
UKURAN_FITUR = config.MAKS_TANGAN * UKURAN_TANGAN + 1  # 126 + 1 penanda = 127
INDEKS_PERGELANGAN = 0  # landmark 0 = pergelangan tangan (wrist)


# ----------------------------------------------------------------------
# Bagian 1: hitungan murni (tanpa MediaPipe, mudah diuji)
# ----------------------------------------------------------------------

def urutkan_tangan(daftar_tangan):
    """Urutkan tangan dari kiri ke kanan di layar berdasarkan x pergelangan.

    Kenapa bukan label Left/Right MediaPipe? Label itu bisa tertukar saat
    gambar dicerminkan, sehingga fitur training dan real-time jadi berbeda.
    """
    return sorted(daftar_tangan, key=lambda tangan: tangan[INDEKS_PERGELANGAN][0])


def normalisasi_bersama(daftar_tangan):
    """Normalisasi SEMUA titik dari semua tangan sekaligus.

    Contoh angka sederhana (2 titik saja, 1 dimensi biar mudah dibayangkan):
      titik = [0.4, 0.8]
      1) pusat = rata-rata = 0.6
      2) dikurangi pusat   = [-0.2, +0.2]
      3) jarak maksimum    = 0.2
      4) dibagi skala      = [-1, +1]
    Hasilnya tidak peduli tangan berada di pojok mana atau sebesar apa di
    kamera. Karena pusat dan skala dihitung untuk SEMUA titik bersama, jarak
    relatif antara tangan kiri dan kanan tetap terjaga (penting untuk huruf
    dua tangan).
    """
    semua_titik = np.vstack(daftar_tangan)          # bentuk (jumlah_titik, 3)
    pusat = semua_titik.mean(axis=0)                 # rata-rata x, y, z
    titik_geser = semua_titik - pusat
    jarak = np.linalg.norm(titik_geser, axis=1)      # jarak tiap titik ke pusat
    skala = jarak.max()
    if skala < 1e-9:                                 # hindari bagi nol
        skala = 1.0
    titik_normal = titik_geser / skala

    # Pecah kembali menjadi per tangan
    hasil = []
    awal = 0
    for tangan in daftar_tangan:
        akhir = awal + len(tangan)
        hasil.append(titik_normal[awal:akhir])
        awal = akhir
    return hasil


def landmark_ke_fitur(daftar_tangan):
    """Ubah landmark (list array 21x3 per tangan) menjadi vektor fitur tetap.

    Kembalikan None jika tidak ada tangan.
    """
    if not daftar_tangan:
        return None

    daftar_tangan = [np.asarray(t, dtype=np.float64) for t in daftar_tangan]
    daftar_tangan = urutkan_tangan(daftar_tangan)[: config.MAKS_TANGAN]
    daftar_tangan = normalisasi_bersama(daftar_tangan)

    jumlah_tangan = len(daftar_tangan)
    vektor = np.zeros(UKURAN_FITUR, dtype=np.float32)
    for nomor, tangan in enumerate(daftar_tangan):
        awal = nomor * UKURAN_TANGAN
        vektor[awal:awal + UKURAN_TANGAN] = tangan.flatten()
    # Slot tangan ke-2 sudah nol jika hanya 1 tangan (padding).
    vektor[-1] = jumlah_tangan  # fitur penanda jumlah tangan
    return vektor


def nama_kolom_fitur():
    """Nama kolom untuk CSV: t1_x0, t1_y0, t1_z0, ... t2_z20, jumlah_tangan."""
    kolom = []
    for nomor_tangan in (1, 2):
        for titik in range(TITIK_PER_TANGAN):
            for sumbu in ("x", "y", "z"):
                kolom.append(f"t{nomor_tangan}_{sumbu}{titik}")
    kolom.append("jumlah_tangan")
    return kolom


def cerminkan_fitur(matriks_fitur):
    """Buat versi cermin horizontal (flip kiri-kanan) dari matriks fitur (N, 127).

    Kenapa perlu?
    1) Rekaman webcam yang dicerminkan (mirror) vs foto dataset punya arah X
       yang terbalik; tanpa ini, tangan kiri dan kanan tertukar (t1 <-> t2)
       dan akurasi jatuh drastis.
    2) Sekaligus membuat model mengenali penutur kidal maupun tangan kanan.
    """
    hasil = np.array(matriks_fitur, dtype=np.float32, copy=True)
    t1 = hasil[:, :UKURAN_TANGAN].copy()
    t2 = hasil[:, UKURAN_TANGAN : 2 * UKURAN_TANGAN].copy()

    # Balik tanda koordinat x (indeks 0, 3, 6, ..., 60) pada kedua slot tangan
    t1[:, 0::3] *= -1.0
    t2[:, 0::3] *= -1.0

    # Jika 2 tangan, setelah dicerminkan posisi kiri-kanan bertukar (t1 <-> t2)
    dua_tangan = hasil[:, -1] > 1.5
    hasil[:, :UKURAN_TANGAN] = np.where(dua_tangan[:, None], t2, t1)
    hasil[:, UKURAN_TANGAN : 2 * UKURAN_TANGAN] = np.where(dua_tangan[:, None], t1, t2)
    return hasil


# ----------------------------------------------------------------------
# Bagian 2: MediaPipe
# ----------------------------------------------------------------------

def buat_detektor(gambar_statis=False):
    """Buat detektor MediaPipe Hands.

    gambar_statis=True untuk dataset foto (tiap gambar berdiri sendiri),
    False untuk video (MediaPipe melacak tangan antar frame, lebih cepat).
    Ingat panggil detektor.close() setelah selesai.
    """
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "MediaPipe tidak bisa dimuat atau versinya tidak punya mp.solutions. "
            "Pastikan `pip install mediapipe` berhasil dan Python 3.10/3.11. "
            f"Detail: {error}"
        )
    return mp_hands.Hands(
        static_image_mode=gambar_statis,
        max_num_hands=config.MAKS_TANGAN,
        min_detection_confidence=config.MIN_KEYAKINAN_DETEKSI,
        min_tracking_confidence=config.MIN_KEYAKINAN_TRACKING,
    )


def deteksi_landmark(gambar_bgr, detektor):
    """Jalankan MediaPipe pada satu gambar BGR (format OpenCV).

    Kembalikan list array (21, 3) per tangan, atau list kosong jika tidak ada.
    Koordinat x,y MediaPipe sudah berskala 0-1 terhadap lebar/tinggi gambar.
    """
    import cv2

    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe butuh RGB
    hasil = detektor.process(gambar_rgb)
    if not hasil.multi_hand_landmarks:
        return []
    daftar_tangan = []
    for tangan in hasil.multi_hand_landmarks:
        titik = np.array([[p.x, p.y, p.z] for p in tangan.landmark], dtype=np.float64)
        daftar_tangan.append(titik)
    return daftar_tangan


def siapkan_frame_untuk_deteksi(frame_bgr):
    """Tentukan frame yang dikirim ke detektor (cermin atau asli).

    Pengumpulan data dan real-time WAJIB memakai fungsi ini supaya orientasi
    kiri-kanan sama; kalau tidak, fitur training dan real-time akan berbeda.
    """
    import cv2

    if config.FITUR_DARI_FRAME_CERMIN:
        return cv2.flip(frame_bgr, 1)
    return frame_bgr


def ekstrak_fitur(gambar_bgr, detektor):
    """Fungsi utama: gambar -> vektor fitur (127 angka) atau None."""
    return landmark_ke_fitur(deteksi_landmark(gambar_bgr, detektor))


def ekstrak_fitur_dan_landmark(gambar_bgr, detektor):
    """Sama seperti ekstrak_fitur, tapi juga mengembalikan landmark mentah.

    Berguna untuk menggambar overlay di real-time. Hasil: (vektor|None, landmark).
    """
    landmark = deteksi_landmark(gambar_bgr, detektor)
    return landmark_ke_fitur(landmark), landmark
