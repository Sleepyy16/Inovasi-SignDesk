"""
SignDesk - src/uji_features.py
==============================
Skrip pengujian untuk src/features.py.

Menjalankan serangkaian uji tanpa membutuhkan gambar nyata,
sehingga bisa dijalankan meski belum punya dataset.

Cara menjalankan:
    Windows:  .venv\\Scripts\\activate  lalu  python src/uji_features.py
    Mac/Linux: source .venv/bin/activate  lalu  python src/uji_features.py

Jika semua uji LULUS, siap lanjut ke T2.
"""

import sys
import os
import time

import numpy as np

# Tambahkan root proyek ke path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import config
from src.features import (
    buat_detektor,
    ekstrak_fitur,
    gambar_overlay,
    nama_kolom_fitur,
    _urutkan_tangan,
    _normalisasi_bersama,
    _susun_vektor,
)

# ──────────────────────────────────────────────────────────────────
# Fungsi pembantu uji
# ──────────────────────────────────────────────────────────────────

_lulus = 0
_gagal = 0


def cek(kondisi: bool, nama_uji: str, detail: str = ""):
    """Cetak hasil satu uji."""
    global _lulus, _gagal
    if kondisi:
        _lulus += 1
        print(f"  ✅ LULUS  {nama_uji}")
    else:
        _gagal += 1
        print(f"  ❌ GAGAL  {nama_uji}")
        if detail:
            print(f"           → {detail}")


def buat_tangan_palsu(x_pergelangan: float = 0.5) -> np.ndarray:
    """
    Membuat data landmark tangan palsu (untuk pengujian tanpa kamera).

    Menghasilkan 21 titik acak dengan x pergelangan yang bisa diatur,
    sehingga kita bisa menguji pengurutan tangan.

    Parameters
    ----------
    x_pergelangan : float
        Posisi x pergelangan (landmark ke-0). Nilai 0.0–1.0.
    """
    np.random.seed(42)
    tangan = np.random.rand(21, 3).astype(np.float32)
    tangan[0, 0] = x_pergelangan  # landmark 0 = WRIST, komponen 0 = x
    return tangan


# ──────────────────────────────────────────────────────────────────
# UJI 1: Nama kolom fitur
# ──────────────────────────────────────────────────────────────────

def uji_nama_kolom():
    print("\n📋 Uji 1: nama_kolom_fitur()")
    kolom = nama_kolom_fitur()

    cek(len(kolom) == config.UKURAN_VEKTOR_FITUR,
        f"Jumlah kolom = {config.UKURAN_VEKTOR_FITUR}",
        f"Dapat: {len(kolom)}")

    cek(kolom[0] == "f_000",
        "Kolom pertama = 'f_000'",
        f"Dapat: {kolom[0]}")

    cek(kolom[-1] == "f_126",
        "Kolom terakhir = 'f_126'",
        f"Dapat: {kolom[-1]}")

    print(f"     Contoh kolom: {kolom[:5]} ... {kolom[-3:]}")


# ──────────────────────────────────────────────────────────────────
# UJI 2: Pengurutan tangan berdasarkan posisi x
# ──────────────────────────────────────────────────────────────────

def uji_pengurutan_tangan():
    print("\n📋 Uji 2: _urutkan_tangan()")

    # Tangan A di x=0.8 (kanan layar), tangan B di x=0.2 (kiri layar)
    tangan_a = buat_tangan_palsu(x_pergelangan=0.8)
    tangan_b = buat_tangan_palsu(x_pergelangan=0.2)

    # Input sengaja terbalik (A dulu, B kemudian)
    hasil = _urutkan_tangan([tangan_a, tangan_b])

    cek(len(hasil) == 2,
        "Jumlah tangan tetap 2")

    cek(float(hasil[0][0, 0]) < float(hasil[1][0, 0]),
        "Tangan kiri (x kecil) ada di posisi pertama",
        f"x[0]={hasil[0][0,0]:.2f}, x[1]={hasil[1][0,0]:.2f}")

    cek(abs(float(hasil[0][0, 0]) - 0.2) < 0.001,
        "Tangan pertama adalah tangan_b (x=0.2)",
        f"x pergelangan tangan pertama = {hasil[0][0,0]:.4f}")

    # Uji dengan 1 tangan (tidak boleh crash)
    hasil_1 = _urutkan_tangan([tangan_a])
    cek(len(hasil_1) == 1,
        "Pengurutan 1 tangan tidak crash")


# ──────────────────────────────────────────────────────────────────
# UJI 3: Normalisasi bersama
# ──────────────────────────────────────────────────────────────────

def uji_normalisasi():
    print("\n📋 Uji 3: _normalisasi_bersama()")

    # Kasus sederhana: 1 tangan dengan semua titik diketahui
    tangan_satu = np.array(
        [[0.5, 0.5, 0.0]] * 21,  # semua titik di tengah
        dtype=np.float32
    )
    hasil_satu = _normalisasi_bersama([tangan_satu])

    cek(len(hasil_satu) == 1,
        "Output tetap 1 tangan")

    # Kalau semua titik sama (skala=0), tidak boleh NaN
    cek(not np.any(np.isnan(hasil_satu[0])),
        "Tidak ada NaN meski semua titik identik (skala=0)")

    # Kasus 2 tangan
    tangan_kiri  = buat_tangan_palsu(x_pergelangan=0.2)
    tangan_kanan = buat_tangan_palsu(x_pergelangan=0.7)
    hasil_dua = _normalisasi_bersama([tangan_kiri, tangan_kanan])

    cek(len(hasil_dua) == 2,
        "Output tetap 2 tangan")

    # Setelah normalisasi, nilai absolut terbesar ≤ 1.0
    # (koordinat z bisa > 1 karena hanya x,y yang dipakai untuk skala,
    #  tapi x dan y pasti ≤ 1)
    semua_xy = np.vstack([t[:, :2] for t in hasil_dua])
    maks_xy = np.abs(semua_xy).max()
    cek(maks_xy <= 1.0 + 1e-5,
        f"Nilai x,y ternormalisasi ≤ 1.0 (maks = {maks_xy:.4f})")

    # Pusat harus mendekati nol
    pusat = semua_xy.mean()
    cek(abs(pusat) < 0.1,
        f"Pusat mendekati 0 (pusat = {pusat:.6f})")

    print(f"     Rentang nilai setelah normalisasi: "
          f"[{semua_xy.min():.3f}, {semua_xy.max():.3f}]")


# ──────────────────────────────────────────────────────────────────
# UJI 4: Penyusunan vektor
# ──────────────────────────────────────────────────────────────────

def uji_susun_vektor():
    print("\n📋 Uji 4: _susun_vektor()")

    # --- 4a: 2 tangan ---
    tangan_kiri  = buat_tangan_palsu(0.2)
    tangan_kanan = buat_tangan_palsu(0.7)
    hasil_norm   = _normalisasi_bersama([tangan_kiri, tangan_kanan])
    vektor_dua   = _susun_vektor(hasil_norm)

    cek(vektor_dua.shape == (config.UKURAN_VEKTOR_FITUR,),
        f"Ukuran vektor 2 tangan = {config.UKURAN_VEKTOR_FITUR}",
        f"Dapat: {vektor_dua.shape}")

    cek(vektor_dua[-1] == 2.0,
        "Fitur terakhir = 2.0 (dua tangan)",
        f"Dapat: {vektor_dua[-1]}")

    cek(vektor_dua.dtype == np.float32,
        "Tipe data float32",
        f"Dapat: {vektor_dua.dtype}")

    # --- 4b: 1 tangan saja ---
    hasil_norm_1 = _normalisasi_bersama([tangan_kiri])
    vektor_satu  = _susun_vektor(hasil_norm_1)

    cek(vektor_satu.shape == (config.UKURAN_VEKTOR_FITUR,),
        f"Ukuran vektor 1 tangan = {config.UKURAN_VEKTOR_FITUR}",
        f"Dapat: {vektor_satu.shape}")

    cek(vektor_satu[-1] == 1.0,
        "Fitur terakhir = 1.0 (satu tangan)",
        f"Dapat: {vektor_satu[-1]}")

    # Slot tangan kedua harus nol semua (padding)
    slot_tangan_2 = vektor_satu[config.FITUR_PER_TANGAN: config.FITUR_DUA_TANGAN]
    cek(np.all(slot_tangan_2 == 0.0),
        "Slot tangan kedua = semua nol (padding)")

    # --- 4c: 0 tangan (batas atas: tidak boleh crash) ---
    vektor_nol = _susun_vektor([])
    cek(vektor_nol.shape == (config.UKURAN_VEKTOR_FITUR,),
        "Ukuran vektor 0 tangan = 127 (semua nol + 0.0)")
    cek(vektor_nol[-1] == 0.0,
        "Fitur terakhir = 0.0 (nol tangan)")

    # Cetak tata letak vektor
    print(f"\n     Tata letak vektor fitur ({config.UKURAN_VEKTOR_FITUR} angka):")
    print(f"       [0 … {config.FITUR_PER_TANGAN-1}]"
          f"   = {config.FITUR_PER_TANGAN} fitur tangan kiri "
          f"({config.TITIK_PER_TANGAN} titik × {config.KOMPONEN_PER_TITIK} koordinat)")
    print(f"       [{config.FITUR_PER_TANGAN} … {config.FITUR_DUA_TANGAN-1}]"
          f" = {config.FITUR_PER_TANGAN} fitur tangan kanan (atau nol jika 1 tangan)")
    print(f"       [{config.FITUR_DUA_TANGAN}]"
          f"           = 1  fitur jumlah tangan (1.0 atau 2.0)")
    print(f"       Total = {config.UKURAN_VEKTOR_FITUR}")

    # Cetak beberapa nilai contoh dari vektor 2 tangan
    print(f"\n     Contoh 10 nilai pertama (2 tangan): "
          f"{vektor_dua[:10].round(4)}")
    print(f"     Contoh 5 nilai terakhir (2 tangan):  "
          f"{vektor_dua[-5:].round(4)}")


# ──────────────────────────────────────────────────────────────────
# UJI 5: Konsistensi — vektor yang sama menghasilkan hasil yang sama
# ──────────────────────────────────────────────────────────────────

def uji_konsistensi():
    print("\n📋 Uji 5: Konsistensi (input sama → output sama)")

    tangan = buat_tangan_palsu(0.4)
    hasil_norm_a = _normalisasi_bersama([tangan])
    hasil_norm_b = _normalisasi_bersama([tangan])
    vektor_a = _susun_vektor(hasil_norm_a)
    vektor_b = _susun_vektor(hasil_norm_b)

    cek(np.allclose(vektor_a, vektor_b),
        "Dua pemanggilan dengan input sama menghasilkan vektor identik")


# ──────────────────────────────────────────────────────────────────
# UJI 6: Uji dengan gambar hitam (MediaPipe, tidak ada tangan)
# ──────────────────────────────────────────────────────────────────

def uji_gambar_hitam():
    """
    Menguji ekstrak_fitur() dengan gambar hitam (tidak ada tangan).
    Harus mengembalikan None tanpa crash.
    """
    print("\n📋 Uji 6: ekstrak_fitur() dengan gambar tanpa tangan")

    try:
        import cv2
    except ImportError:
        print("  ⚠️  OpenCV tidak tersedia, uji 6 dilewati.")
        return

    # Buat frame hitam (tidak ada tangan di dalamnya)
    frame_hitam = np.zeros(
        (config.TINGGI_KAMERA, config.LEBAR_KAMERA, 3),
        dtype=np.uint8
    )

    waktu_mulai = time.time()
    with buat_detektor(mode_gambar_statis=True) as detektor:
        fitur, hasil = ekstrak_fitur(frame_hitam, detektor)
    waktu_selesai = time.time()

    cek(fitur is None,
        "Tidak ada tangan terdeteksi → fitur = None")

    cek(hasil is not None,
        "Objek hasil MediaPipe tetap dikembalikan (bukan None)")

    waktu_ms = (waktu_selesai - waktu_mulai) * 1000
    cek(waktu_ms < 500,
        f"Waktu proses < 500ms (aktual: {waktu_ms:.1f}ms)")

    print(f"     Waktu proses: {waktu_ms:.1f} ms")


# ──────────────────────────────────────────────────────────────────
# UJI 7: Uji buat_detektor() (buka dan tutup dengan 'with')
# ──────────────────────────────────────────────────────────────────

def uji_buat_detektor():
    print("\n📋 Uji 7: buat_detektor()")

    # Uji mode gambar statis
    try:
        d = buat_detektor(mode_gambar_statis=True)
        d.close()
        cek(True, "mode_gambar_statis=True berhasil dibuat dan ditutup")
    except Exception as e:
        cek(False, "mode_gambar_statis=True", str(e))

    # Uji mode video (real-time)
    try:
        with buat_detektor(mode_gambar_statis=False) as d:
            pass
        cek(True, "mode_gambar_statis=False dengan 'with' berhasil")
    except Exception as e:
        cek(False, "mode_gambar_statis=False dengan 'with'", str(e))


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SignDesk — Uji src/features.py")
    print("=" * 60)

    uji_nama_kolom()
    uji_buat_detektor()
    uji_pengurutan_tangan()
    uji_normalisasi()
    uji_susun_vektor()
    uji_konsistensi()
    uji_gambar_hitam()

    print("\n" + "=" * 60)
    print(f"  Hasil: {_lulus} LULUS, {_gagal} GAGAL")
    print("=" * 60)

    if _gagal == 0:
        print("\n🎉 Semua uji lulus! src/features.py siap dipakai.")
        print("   Lanjut ke T2: src/extract_from_images.py")
    else:
        print(f"\n⚠️  Ada {_gagal} uji yang gagal.")
        print("   Periksa pesan error di atas, lalu perbaiki features.py.")

    return _gagal == 0


if __name__ == "__main__":
    sukses = main()
    sys.exit(0 if sukses else 1)

