"""Uji features.py.

Dua tahap:
  1) Uji hitungan dengan data buatan (tanpa MediaPipe/kamera).
  2) Jika diberi gambar, uji deteksi tangan sungguhan.

Cara pakai (dari folder utama proyek):
  python src/uji_features.py
  python src/uji_features.py path/ke/gambar_tangan.jpg
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

import features


def tangan_buatan(geser_x, skala=1.0):
    """Buat 21 titik acak (tapi tetap) sebagai tangan palsu."""
    acak = np.random.default_rng(0)
    titik = acak.random((21, 3)) * skala
    titik[:, 0] += geser_x
    return titik


def uji_data_buatan():
    print("=== Tahap 1: data buatan ===")

    # Satu tangan
    v1 = features.landmark_ke_fitur([tangan_buatan(0.3)])
    assert v1.shape == (features.UKURAN_FITUR,), "ukuran vektor salah"
    assert v1[-1] == 1, "penanda jumlah tangan harus 1"
    assert np.all(v1[features.UKURAN_TANGAN:-1] == 0), "slot tangan ke-2 harus nol"
    print("1 tangan   : ukuran", v1.shape, "| penanda =", v1[-1])

    # Dua tangan
    v2 = features.landmark_ke_fitur([tangan_buatan(0.7), tangan_buatan(0.1)])
    assert v2[-1] == 2
    print("2 tangan   : ukuran", v2.shape, "| penanda =", v2[-1])

    # Urutan input tidak boleh berpengaruh (diurutkan berdasarkan x)
    v2_terbalik = features.landmark_ke_fitur([tangan_buatan(0.1), tangan_buatan(0.7)])
    assert np.allclose(v2, v2_terbalik), "urutan input seharusnya tidak berpengaruh"
    print("urutan tangan tidak berpengaruh: OK")

    # Geser dan perbesar tangan tidak boleh mengubah fitur
    asli = features.landmark_ke_fitur([tangan_buatan(0.2)])
    geser = features.landmark_ke_fitur([tangan_buatan(0.2) + 5.0])
    besar = features.landmark_ke_fitur([tangan_buatan(0.2) * 3.0])
    assert np.allclose(asli, geser, atol=1e-5), "fitur harus tahan pergeseran"
    assert np.allclose(asli, besar, atol=1e-5), "fitur harus tahan perbesaran"
    print("tahan geser dan skala: OK")

    # Tanpa tangan
    assert features.landmark_ke_fitur([]) is None
    print("tanpa tangan -> None: OK")

    print("Beberapa nilai (1 tangan, 6 angka pertama):", np.round(v1[:6], 3))
    print("Jumlah kolom CSV:", len(features.nama_kolom_fitur()), "(harus 127)")
    assert len(features.nama_kolom_fitur()) == features.UKURAN_FITUR


def uji_gambar(path_gambar):
    print("\n=== Tahap 2: gambar sungguhan ===")
    import cv2

    path = Path(path_gambar)
    if not path.exists():
        print(f"File tidak ditemukan: {path}")
        return
    # np.fromfile + imdecode supaya path berspasi/non-ASCII aman di Windows
    data = np.fromfile(str(path), dtype=np.uint8)
    gambar = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if gambar is None:
        print("File bukan gambar yang valid.")
        return

    detektor = features.buat_detektor(gambar_statis=True)
    try:
        vektor = features.ekstrak_fitur(gambar, detektor)
    finally:
        detektor.close()

    if vektor is None:
        print("Tangan tidak terdeteksi. Coba gambar lain (tangan jelas, cahaya cukup).")
        return
    print("Ukuran vektor:", vektor.shape)
    print("Jumlah tangan:", int(vektor[-1]))
    print("10 nilai pertama:", np.round(vektor[:10], 3))


if __name__ == "__main__":
    uji_data_buatan()
    if len(sys.argv) > 1:
        uji_gambar(sys.argv[1])
    else:
        print("\n(Tambahkan path gambar untuk menguji MediaPipe: "
              "python src/uji_features.py gambar.jpg)")
