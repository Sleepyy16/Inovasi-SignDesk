"""Rekam data landmark sendiri lewat webcam.

PRIVASI: yang disimpan HANYA angka landmark + label + person_id.
Tidak ada gambar, video, atau wajah yang disimpan ke disk.

Kontrol (di jendela video):
  - Tekan huruf A-Z       : pilih huruf (KECUALI Q, karena Q = keluar)
  - Panah kiri/kanan atau [ dan ] : pilih huruf sebelumnya/berikutnya
    (cara ini juga dipakai untuk memilih huruf Q)
  - SPASI                 : rekam sampel (tahan untuk rekam beberapa per detik)
  - Q                     : simpan lalu keluar

Data DITAMBAHKAN (append) ke data/landmarks/landmarks_webcam.csv, tidak menimpa.

Jalankan dari folder utama:  python src/collect_data.py
"""

import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd

import config
import features

PESAN_PERSETUJUAN = """
=== PERSETUJUAN PEROLEHAN DATA ===
Program ini merekam posisi titik-titik tangan (angka landmark) dari kamera.
- Yang disimpan: angka landmark, huruf (label), dan kode penutur (person_id).
- TIDAK disimpan: gambar, video, wajah, atau nama asli.
- Data dipakai untuk melatih model pengenal alfabet BISINDO (proyek SignDesk).
Pastikan penutur sudah memberi persetujuan (idealnya tertulis) sebelum merekam.
"""

PANDUAN_VARIASI = [
    "Variasikan tiap sesi: jarak ke kamera,",
    "sudut tangan, cahaya, dan posisi tangan di layar.",
]


def minta_persetujuan_dan_id():
    print(PESAN_PERSETUJUAN)
    jawaban = input("Ketik SETUJU untuk melanjutkan: ").strip().upper()
    if jawaban != "SETUJU":
        print("Tidak ada persetujuan. Program berhenti.")
        return None
    while True:
        person_id = input("Masukkan person_id (contoh P01, tanpa nama asli): ").strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{1,20}", person_id):
            return person_id
        print("person_id hanya boleh huruf/angka/_/-, maksimal 20 karakter.")


def hitung_data_lama():
    """Hitung sampel per huruf yang sudah ada di CSV (untuk ditampilkan)."""
    hitungan = {h: 0 for h in config.HURUF}
    if config.FILE_CSV_WEBCAM.exists():
        try:
            lama = pd.read_csv(config.FILE_CSV_WEBCAM, usecols=["label"])
            for label, jumlah in lama["label"].value_counts().items():
                if label in hitungan:
                    hitungan[label] = int(jumlah)
        except Exception as error:  # file rusak jangan menghentikan program
            print(f"PERINGATAN: gagal membaca CSV lama ({error}). Data baru tetap ditambahkan.")
    return hitungan


def baca_tombol(kode, indeks_huruf):
    """Ubah kode tombol menjadi (indeks_huruf_baru, aksi). aksi: 'rekam', 'keluar', None."""
    if kode == -1:
        return indeks_huruf, None
    total = len(config.HURUF)
    if kode in (ord("q"), ord("Q")):
        return indeks_huruf, "keluar"
    if kode == 32:
        return indeks_huruf, "rekam"
    if kode == ord("["):
        return (indeks_huruf - 1) % total, None
    if kode == ord("]"):
        return (indeks_huruf + 1) % total, None
    if kode in (2424832, 65361):    # panah kiri (Windows, Linux)
        return (indeks_huruf - 1) % total, None
    if kode in (2555904, 65363):    # panah kanan (Windows, Linux)
        return (indeks_huruf + 1) % total, None
    if ord("a") <= kode <= ord("z") or ord("A") <= kode <= ord("Z"):
        huruf = chr(kode).upper()
        return config.HURUF.index(huruf), None
    return indeks_huruf, None


def gambar_landmark(frame, daftar_tangan):
    """Gambar titik landmark di frame (koordinat 0-1 -> piksel)."""
    tinggi, lebar = frame.shape[:2]
    for tangan in daftar_tangan:
        for x, y, _ in tangan:
            cv2.circle(frame, (int(x * lebar), int(y * tinggi)), 3, (0, 255, 0), -1)


def gambar_info(frame, huruf, hitungan_sesi, hitungan_lama, ada_tangan, person_id):
    total_huruf = hitungan_sesi[huruf] + hitungan_lama[huruf]
    cv2.putText(frame, f"Huruf: {huruf}  (sesi: {hitungan_sesi[huruf]}, total: {total_huruf})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    status = "Tangan terdeteksi" if ada_tangan else "Tangan TIDAK terdeteksi"
    warna = (0, 255, 0) if ada_tangan else (0, 0, 255)
    cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)
    cv2.putText(frame, f"Penutur: {person_id} | Total sesi: {sum(hitungan_sesi.values())}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tinggi = frame.shape[0]
    cv2.putText(frame, "Huruf/panah: pilih | SPASI: rekam | Q: simpan & keluar",
                (10, tinggi - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for nomor, baris in enumerate(PANDUAN_VARIASI):
        cv2.putText(frame, baris, (10, tinggi - 30 + nomor * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)


def simpan_csv(baris, person_id):
    """Simpan baris ke CSV; tawarkan ganti (timpa) jika huruf untuk person_id sudah ada."""
    if not baris:
        print("Tidak ada sampel baru untuk disimpan.")
        return
    config.LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    kolom = ["label", "person_id", "source"] + features.nama_kolom_fitur()
    tabel_baru = pd.DataFrame(baris, columns=kolom)

    if config.FILE_CSV_WEBCAM.exists():
        try:
            tabel_lama = pd.read_csv(config.FILE_CSV_WEBCAM)
            if "f_000" in tabel_lama.columns and len(tabel_lama.columns) == len(kolom):
                tabel_lama.columns = kolom
            huruf_direkam = sorted(tabel_baru["label"].unique())
            mask_bentrok = (
                (tabel_lama["person_id"].astype(str) == str(person_id))
                & (tabel_lama["label"].isin(huruf_direkam))
            )
            jumlah_bentrok = int(mask_bentrok.sum())
            if jumlah_bentrok > 0:
                daftar = ", ".join(huruf_direkam)
                jawab = input(
                    f"Data lama untuk person_id '{person_id}' pada huruf [{daftar}] "
                    f"ditemukan ({jumlah_bentrok} baris). Ganti/timpa dengan rekaman baru? [Y/n]: "
                ).strip().lower()
                if jawab in ("", "y", "ya", "yes"):
                    tabel_lama = tabel_lama[~mask_bentrok].reset_index(drop=True)
                    gabungan = pd.concat([tabel_lama, tabel_baru], ignore_index=True)
                    gabungan.to_csv(config.FILE_CSV_WEBCAM, index=False)
                    print(
                        f"Data lama [{daftar}] milik {person_id} diganti dengan "
                        f"{len(baris)} sampel baru di {config.FILE_CSV_WEBCAM}"
                    )
                    return
        except Exception as error:
            print(f"PERINGATAN: gagal memeriksa data lama ({error}), lanjut append.")

    file_baru = not config.FILE_CSV_WEBCAM.exists()
    tabel_baru.to_csv(config.FILE_CSV_WEBCAM, mode="a", header=file_baru, index=False)
    print(f"{len(baris)} sampel ditambahkan ke {config.FILE_CSV_WEBCAM}")


def main():
    person_id = minta_persetujuan_dan_id()
    if person_id is None:
        return

    kamera = cv2.VideoCapture(config.INDEKS_KAMERA)
    if not kamera.isOpened():
        print(f"Kamera indeks {config.INDEKS_KAMERA} tidak bisa dibuka.")
        print("Tutup aplikasi lain yang memakai kamera, cek izin kamera, atau ubah INDEKS_KAMERA di config.py.")
        return
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, config.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TINGGI_KAMERA)

    detektor = features.buat_detektor(gambar_statis=False)
    hitungan_lama = hitung_data_lama()
    hitungan_sesi = {h: 0 for h in config.HURUF}
    baris = []              # sampel yang belum disimpan
    indeks_huruf = 0
    waktu_rekam_terakhir = 0.0
    pesan = ""
    waktu_pesan = 0.0

    print("Kamera siap. Fokus ke jendela video, lalu mulai merekam. Tekan Q untuk keluar.")
    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                print("Gagal membaca frame dari kamera.")
                break

            frame = features.siapkan_frame_untuk_deteksi(frame)
            vektor, landmark = features.ekstrak_fitur_dan_landmark(frame, detektor)
            gambar_landmark(frame, landmark)
            if config.CERMINKAN_KAMERA and not config.FITUR_DARI_FRAME_CERMIN:
                frame = cv2.flip(frame, 1)  # mirror hanya untuk tampilan

            huruf = config.HURUF[indeks_huruf]
            gambar_info(frame, huruf, hitungan_sesi, hitungan_lama, vektor is not None, person_id)
            if pesan and time.time() - waktu_pesan < 1.5:
                cv2.putText(frame, pesan, (10, 115), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 165, 255), 2)
            cv2.imshow("Kumpulkan Data SignDesk", frame)

            kode = cv2.waitKeyEx(1)
            indeks_huruf, aksi = baca_tombol(kode, indeks_huruf)
            if aksi == "keluar":
                break
            if aksi == "rekam":
                sekarang = time.time()
                if vektor is None:
                    pesan, waktu_pesan = "Tangan tidak terdeteksi, tidak direkam", sekarang
                elif hitungan_sesi[huruf] >= config.MAKS_SAMPEL_PER_HURUF_SESI:
                    pesan, waktu_pesan = "Batas sesi huruf ini tercapai. Ganti huruf/variasi.", sekarang
                elif sekarang - waktu_rekam_terakhir >= config.JEDA_REKAM_DETIK:
                    baris.append([huruf, person_id, "webcam", *vektor])
                    hitungan_sesi[huruf] += 1
                    waktu_rekam_terakhir = sekarang
    finally:
        # Simpan walau terjadi error, supaya rekaman tidak hilang.
        kamera.release()
        detektor.close()
        cv2.destroyAllWindows()
        simpan_csv(baris, person_id)


if __name__ == "__main__":
    main()
