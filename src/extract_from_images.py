"""Ekstraksi landmark dari dataset gambar di data/raw_images/.

Struktur dataset (satu folder per DATASET, lalu satu sub-folder per huruf):
    data/raw_images/kaggle/A/foto1.jpg
    data/raw_images/mendeley/B/foto7.png
    ...
Nama folder dataset dipakai sebagai person_id (PROKSI, bukan penutur sebenarnya):
split berdasarkan dataset tetap lebih jujur daripada split acak, tapi belum
sama dengan split per penutur. Jika nama file memuat id penutur, atur
POLA_PERSON_ID di config.py; id itu akan dipakai (digabung: dataset_id).
Jika datasetmu berbeda:
  - Folder huruf bukan A-Z (mis. "kelas_01"): ganti nama foldernya atau ubah
    fungsi `label_dari_folder`.

Yang disimpan ke CSV hanya angka landmark + label + person_id + nama sumber file.
Gambar TIDAK disalin atau diubah.

Jalankan dari folder utama:  python src/extract_from_images.py
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

import config
import features


def label_dari_folder(folder):
    """Nama folder = label huruf. Kembalikan None jika bukan huruf A-Z."""
    nama = folder.name.strip().upper()
    return nama if nama in config.HURUF else None


def tebak_person_id(nama_file):
    """Ambil person_id dari nama file memakai POLA_PERSON_ID, jika diatur."""
    if config.POLA_PERSON_ID:
        cocok = re.search(config.POLA_PERSON_ID, nama_file)
        if cocok:
            return cocok.group(1)
    return "unknown"


def baca_gambar(path):
    """Baca gambar; np.fromfile agar path berspasi/non-ASCII aman di Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def daftar_gambar(folder):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in config.EKSTENSI_GAMBAR)


def cetak_progress(sekarang, total, awalan=""):
    lebar = 30
    isi = int(lebar * sekarang / max(total, 1))
    bar = "#" * isi + "-" * (lebar - isi)
    print(f"\r{awalan}[{bar}] {sekarang}/{total}", end="", flush=True)


def proses_dataset():
    if not config.RAW_IMAGES_DIR.exists():
        print(f"Folder tidak ada: {config.RAW_IMAGES_DIR}")
        return

    # Kumpulkan semua gambar: raw_images/<dataset>/<HURUF>/gambar
    tugas = []  # (path, label, nama_dataset)
    for folder_dataset in sorted(p for p in config.RAW_IMAGES_DIR.iterdir() if p.is_dir()):
        if label_dari_folder(folder_dataset) is not None and len(folder_dataset.name) == 1:
            print(f"PERINGATAN: '{folder_dataset.name}' tampak folder huruf langsung di raw_images/.")
            print("Susun ulang menjadi raw_images/<nama_dataset>/<HURUF>/. Folder ini dilewati.")
            continue
        for folder in sorted(p for p in folder_dataset.iterdir() if p.is_dir()):
            label = label_dari_folder(folder)
            if label is None:
                print(f"PERINGATAN: folder '{folder_dataset.name}/{folder.name}' dilewati (bukan huruf A-Z).")
                continue
            tugas.extend((gambar, label, folder_dataset.name) for gambar in daftar_gambar(folder))

    if not tugas:
        print("Tidak ada gambar ditemukan.")
        print(f"Susun dataset di {config.RAW_IMAGES_DIR} sebagai <nama_dataset>/<HURUF>/gambar.")
        return

    jumlah = {h: 0 for h in config.HURUF}       # berhasil
    gagal = {h: 0 for h in config.HURUF}        # tangan tidak terdeteksi / gambar rusak
    baris = []

    detektor = features.buat_detektor(gambar_statis=True)
    try:
        for nomor, (path, label, dataset) in enumerate(tugas, start=1):
            cetak_progress(nomor, len(tugas), "Memproses ")
            gambar = baca_gambar(path)
            if gambar is None:
                gagal[label] += 1
                continue
            vektor = features.ekstrak_fitur(gambar, detektor)
            if vektor is None:
                gagal[label] += 1
                continue
            jumlah[label] += 1
            id_file = tebak_person_id(path.name)
            # Dataset dipakai sebagai proksi penutur; gabungkan jika id file tersedia.
            person_id = dataset if id_file == "unknown" else f"{dataset}_{id_file}"
            baris.append([label, person_id, f"{dataset}/{path.parent.name}/{path.name}", *vektor])
    finally:
        detektor.close()
    print()

    if not baris:
        print("Tidak ada satu pun tangan terdeteksi. Cek gambar dan instalasi MediaPipe.")
        return

    kolom = ["label", "person_id", "source"] + features.nama_kolom_fitur()
    tabel = pd.DataFrame(baris, columns=kolom)
    config.LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(config.FILE_CSV_GAMBAR, index=False)

    cetak_laporan(jumlah, gagal, tabel)
    print(f"\nCSV disimpan: {config.FILE_CSV_GAMBAR}")


def cetak_laporan(jumlah, gagal, tabel):
    print("\nHuruf | berhasil | gagal terdeteksi")
    print("------+----------+-----------------")
    for huruf in config.HURUF:
        print(f"  {huruf}   | {jumlah[huruf]:8d} | {gagal[huruf]:8d}")
    print(f"Total berhasil: {sum(jumlah.values())}, gagal: {sum(gagal.values())}")

    ada_data = {h: n for h, n in jumlah.items()}
    kosong = [h for h, n in ada_data.items() if n == 0]
    if kosong:
        print(f"PERINGATAN: huruf tanpa data sama sekali: {', '.join(kosong)}")
    paling_sedikit = min((h for h in config.HURUF if jumlah[h] > 0),
                         key=lambda h: jumlah[h], default=None)
    if paling_sedikit:
        print(f"Huruf dengan data paling sedikit: {paling_sedikit} ({jumlah[paling_sedikit]} sampel)")

    print("\nSampel per person_id (proksi: nama dataset jika id penutur tidak ada):")
    print(tabel["person_id"].value_counts().to_string())
    if tabel["person_id"].nunique() < 2:
        print("\nPERINGATAN: hanya 1 person_id/dataset. Split per penutur belum mungkin;")
        print("evaluasi bisa terlalu optimis. Tambah dataset lain atau data sendiri (T3).")
    else:
        print("\nCATATAN: person_id berupa nama dataset (proksi), BUKAN penutur sebenarnya.")
        print("Satu dataset bisa berisi banyak orang; laporkan ini apa adanya di proposal.")


if __name__ == "__main__":
    proses_dataset()
