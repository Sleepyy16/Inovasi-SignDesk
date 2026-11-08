"""
SignDesk - src/cek_kamera.py
============================
Skrip sederhana untuk memastikan kamera dan OpenCV berfungsi
sebelum mulai mengerjakan bagian lain proyek.

Cara menjalankan:
    Windows:  .venv\\Scripts\\activate  lalu  python src/cek_kamera.py
    Mac/Linux: source .venv/bin/activate  lalu  python src/cek_kamera.py

Tekan Q untuk keluar.
"""

import sys
import os

# Tambahkan root proyek ke path agar bisa import config
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

try:
    import cv2
except ImportError:
    print("❌ OpenCV belum terpasang.")
    print("   Jalankan: pip install opencv-python")
    sys.exit(1)

try:
    import config
except ImportError:
    print("❌ Tidak bisa menemukan config.py.")
    print("   Pastikan kamu menjalankan skrip ini dari folder root proyek (signdesk/).")
    sys.exit(1)


def cek_kamera():
    """
    Membuka webcam dan menampilkan video secara langsung.
    Juga menampilkan informasi teknis kamera di layar.
    """
    print(f"⏳ Mencoba membuka kamera dengan indeks {config.INDEKS_KAMERA}...")
    print("   (Jika kamera tidak terbuka, coba ganti INDEKS_KAMERA di config.py)")

    # Buka kamera
    kamera = cv2.VideoCapture(config.INDEKS_KAMERA)

    # Cek apakah kamera berhasil dibuka
    if not kamera.isOpened():
        print(f"❌ Kamera dengan indeks {config.INDEKS_KAMERA} tidak bisa dibuka.")
        print("   Kemungkinan penyebab:")
        print("   1. Kamera tidak terhubung atau dipakai program lain.")
        print("   2. Indeks kamera salah → coba INDEKS_KAMERA = 1 atau 2 di config.py.")
        print("   3. Driver kamera belum terpasang.")
        return

    # Atur resolusi sesuai config
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, config.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TINGGI_KAMERA)

    print("✅ Kamera berhasil dibuka!")
    print("   Tekan Q pada jendela video untuk keluar.\n")

    jumlah_frame = 0

    while True:
        # Baca satu frame dari kamera
        berhasil, frame = kamera.read()

        if not berhasil:
            print("⚠️  Gagal membaca frame dari kamera. Mungkin koneksi terputus.")
            break

        jumlah_frame += 1

        # Cerminkan tampilan (seperti kaca) agar lebih natural untuk pengguna
        frame_cermin = cv2.flip(frame, 1)

        # Ambil info resolusi aktual (bisa berbeda dari yang diminta)
        tinggi_aktual, lebar_aktual = frame_cermin.shape[:2]

        # Tampilkan informasi di layar
        teks_resolusi = f"Resolusi: {lebar_aktual} x {tinggi_aktual}"
        teks_frame    = f"Frame ke: {jumlah_frame}"
        teks_keluar   = "Tekan Q untuk keluar"

        # Fungsi pembantu: tulis teks dengan latar hitam agar mudah dibaca
        def tulis_teks(frame, teks, posisi, warna=(0, 255, 0)):
            # Latar gelap dulu (supaya teks terbaca di berbagai latar)
            cv2.putText(frame, teks, posisi,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            # Teks berwarna di atas latar
            cv2.putText(frame, teks, posisi,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 1, cv2.LINE_AA)

        tulis_teks(frame_cermin, teks_resolusi, (10, 30))
        tulis_teks(frame_cermin, teks_frame,    (10, 60))
        tulis_teks(frame_cermin, teks_keluar,   (10, tinggi_aktual - 15), warna=(0, 200, 255))

        # Tampilkan frame
        cv2.imshow("SignDesk - Cek Kamera (Tekan Q untuk Keluar)", frame_cermin)

        # Tunggu tombol ditekan; keluar jika Q atau q
        tombol = cv2.waitKey(1) & 0xFF
        if tombol in (ord('q'), ord('Q')):
            print(f"\n✅ Selesai. Total {jumlah_frame} frame ditampilkan.")
            break

    # Bersihkan: lepaskan kamera dan tutup semua jendela
    kamera.release()
    cv2.destroyAllWindows()
    print("   Kamera dan jendela sudah ditutup.")
    print("\n🎉 OpenCV dan kamera berfungsi dengan baik!")
    print("   Lanjut ke T1: buat src/features.py")


if __name__ == "__main__":
    cek_kamera()

