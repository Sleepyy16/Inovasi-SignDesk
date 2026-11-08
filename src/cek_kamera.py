"""Cek apakah webcam dan OpenCV berfungsi. Tekan Q untuk keluar.

Jalankan dari folder utama proyek:  python src/cek_kamera.py
Video hanya ditampilkan, TIDAK disimpan ke disk.
"""

import sys
from pathlib import Path

# Tambahkan folder utama ke path agar `import config` bisa ditemukan
# saat skrip dijalankan dari dalam folder src/.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2

import config


def main():
    kamera = cv2.VideoCapture(config.INDEKS_KAMERA)
    if not kamera.isOpened():
        print(f"Kamera dengan indeks {config.INDEKS_KAMERA} tidak bisa dibuka.")
        print("Coba: tutup aplikasi lain yang memakai kamera (Zoom, Meet),")
        print("cek izin kamera di pengaturan sistem, atau ubah INDEKS_KAMERA di config.py.")
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, config.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TINGGI_KAMERA)
    print("Kamera terbuka. Tekan Q pada jendela video untuk keluar.")

    while True:
        berhasil, frame = kamera.read()
        if not berhasil:
            print("Gagal membaca frame dari kamera.")
            break

        if config.CERMINKAN_KAMERA:
            frame = cv2.flip(frame, 1)  # mirror supaya gerakan terasa natural

        cv2.putText(frame, "Tekan Q untuk keluar", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Cek Kamera SignDesk", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    kamera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
