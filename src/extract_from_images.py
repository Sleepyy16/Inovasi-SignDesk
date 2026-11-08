"""
SignDesk - src/extract_from_images.py
=====================================
Skrip untuk membaca dataset gambar alfabet BISINDO dari folder
data/raw_images/, mengekstrak koordinat landmark tangan menggunakan
fungsi terpusat di src/features.py, lalu menyimpannya dalam format
CSV di data/landmarks/.

STRUKTUR FOLDER YANG DIHARAPKAN:
--------------------------------
data/raw_images/
├── A/
│   ├── gambar_01.jpg
│   └── gambar_02.png
├── B/
│   └── ...
└── Z/

CARA MENYESUAIKAN JIKA STRUKTUR BERBEDA:
----------------------------------------
1. Jika dataset Anda memiliki folder split 'train' dan 'test':
   Jalankan dengan argumen:
     python src/extract_from_images.py --input-dir data/raw_images/train --output landmarks_train.csv
2. Jika nama file memuat ID penutur (misal: "P01_A_01.jpg"):
   Fungsi `tebak_person_id()` di bawah secara otomatis mencoba mengenali
   pola "P01", "person1", "user2", dsb. Jika tidak ditemukan, default adalah "unknown".

HASIL OUTPUT CSV:
-----------------
Kolom: label, person_id, source, f_000, f_001, ..., f_126
Disimpan ke: data/landmarks/landmarks_raw_images.csv (atau sesuai parameter)
"""

import sys
import os
import re
import argparse
import time
from collections import defaultdict

import cv2
import pandas as pd
import numpy as np

# Tambahkan root folder ke sys.path agar bisa import config & src
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import config
from src.features import buat_detektor, ekstrak_fitur, nama_kolom_fitur

# Format ekstensi gambar yang didukung
EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cetak_progress_bar(saat_ini: int, total: int, prefix: str = "", panjang: int = 30):
    """
    Menampilkan progress bar sederhana di konsol tanpa dependensi tambahan.
    """
    if total == 0:
        return
    persen = float(saat_ini) / float(total)
    terisi = int(panjang * persen)
    bar = "=" * terisi + (">" if terisi < panjang else "") + "." * (panjang - terisi - (1 if terisi < panjang else 0))
    sys.stdout.write(f"\r{prefix} [{bar}] {saat_ini}/{total} ({persen * 100:.1f}%)")
    sys.stdout.flush()
    if saat_ini >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def tebak_person_id(nama_file: str) -> str:
    """
    Mencoba menebak person_id dari nama file.
    Contoh:
      - 'P01_A_001.jpg'    -> 'P01'
      - 'person2_B_12.png' -> 'person2'
      - 'subjek_3_C.jpg'   -> 'subjek_3'
      - 'A_123.jpg'        -> 'unknown'
    """
    pola = [
        r"(P\d+)",           # P01, P2, P10
        r"(person[_\-]?\d+)", # person1, person_01
        r"(user[_\-]?\d+)",   # user1, user_02
        r"(subjek[_\-]?\d+)", # subjek1, subjek_03
        r"(speaker[_\-]?\d+)" # speaker1
    ]
    for p in pola:
        match = re.search(p, nama_file, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "unknown"


def ekstrak_dataset(
    folder_input: str,
    file_output: str,
    sumber_data: str = "raw_images",
    person_id_override: str = None
):
    """
    Menelusuri seluruh gambar di folder_input, mengekstrak landmark tangan,
    dan menyimpan hasilnya ke file CSV.
    """
    print("=" * 65)
    print("  SignDesk — Ekstraksi Fitur Landmark dari Dataset Gambar")
    print("=" * 65)
    print(f"📁 Folder Input : {folder_input}")
    print(f"💾 File Output  : {file_output}")
    print(f"🏷️  Sumber Data  : {sumber_data}")
    print("-" * 65)

    if not os.path.exists(folder_input):
        print(f"❌ Error: Folder '{folder_input}' tidak ditemukan.")
        print("   Pastikan folder dataset sudah dibuat dan diletakkan dengan benar.")
        return False

    # 1. Kumpulkan semua file gambar per sub-folder (label)
    daftar_tugas = []
    subfolder_ditemukan = False

    for item in sorted(os.listdir(folder_input)):
        path_item = os.path.join(folder_input, item)
        if os.path.isdir(path_item):
            subfolder_ditemukan = True
            label = item.upper()  # Nama folder dianggap sebagai label huruf
            
            # Ambil semua file gambar di dalam folder ini
            for nama_file in sorted(os.listdir(path_item)):
                _, ext = os.path.splitext(nama_file.lower())
                if ext in EKSTENSI_GAMBAR:
                    path_lengkap = os.path.join(path_item, nama_file)
                    daftar_tugas.append({
                        "label": label,
                        "path": path_lengkap,
                        "nama_file": nama_file
                    })

    if not subfolder_ditemukan or len(daftar_tugas) == 0:
        print(f"⚠️  Tidak ada gambar yang ditemukan di '{folder_input}'.")
        print("\nPetunjuk Penyiapan Dataset:")
        print("1. Buat folder per huruf di dalam:", folder_input)
        print("   Contoh: data/raw_images/A/, data/raw_images/B/, ..., data/raw_images/Z/")
        print("2. Masukkan file gambar (.jpg / .png) ke dalam masing-masing folder huruf.")
        print("3. Jalankan kembali skrip ini: python src/extract_from_images.py\n")
        return False

    total_gambar = len(daftar_tugas)
    print(f"🔍 Ditemukan {total_gambar} gambar yang akan diproses.\n")

    # Siapkan detektor MediaPipe (mode statis untuk akurasi optimal per foto)
    try:
        detektor = buat_detektor(mode_gambar_statis=True)
    except Exception as e:
        print(f"❌ Gagal menginisialisasi MediaPipe HandLandmarker: {e}")
        return False

    # Penampung data hasil ekstraksi
    baris_csv = []
    
    # Statistik per huruf
    total_per_huruf = defaultdict(int)
    gagal_per_huruf = defaultdict(int)
    sukses_per_huruf = defaultdict(int)
    ada_person_id_unknown = False

    waktu_mulai = time.time()

    try:
        for idx, tugas in enumerate(daftar_tugas, start=1):
            label = tugas["label"]
            path_gambar = tugas["path"]
            nama_file = tugas["nama_file"]

            total_per_huruf[label] += 1

            # Tentukan person_id
            if person_id_override:
                pid = person_id_override
            else:
                pid = tebak_person_id(nama_file)
            
            if pid == "unknown":
                ada_person_id_unknown = True

            # Baca gambar menggunakan OpenCV
            img = cv2.imread(path_gambar)
            if img is None:
                gagal_per_huruf[label] += 1
                continue

            # Tentukan sumber data
            if nama_file.startswith("mendeley_"):
                sumber_aktual = "mendeley"
            elif "body" in nama_file.lower() or "wall" in nama_file.lower():
                sumber_aktual = "citra_bisindo"
            else:
                sumber_aktual = sumber_data

            # Ekstrak fitur landmark
            fitur, _ = ekstrak_fitur(img, detektor)

            if fitur is None:
                # Tangan tidak terdeteksi oleh MediaPipe
                gagal_per_huruf[label] += 1
            else:
                sukses_per_huruf[label] += 1
                # Susun satu baris data: [label, person_id, source, f_000, ..., f_126]
                baris = [label, pid, sumber_aktual] + fitur.tolist()
                baris_csv.append(baris)

            # Tampilkan progress bar setiap 5 gambar atau di akhir
            if idx % 5 == 0 or idx == total_gambar:
                cetak_progress_bar(idx, total_gambar, prefix=f"Memproses {label}")

    finally:
        detektor.close()

    durasi = time.time() - waktu_mulai
    total_sukses = len(baris_csv)
    total_gagal = total_gambar - total_sukses

    # 2. Simpan ke CSV
    if total_sukses > 0:
        os.makedirs(os.path.dirname(os.path.abspath(file_output)), exist_ok=True)
        header = ["label", "person_id", "source"] + nama_kolom_fitur()
        df = pd.DataFrame(baris_csv, columns=header)
        df.to_csv(file_output, index=False)
        print(f"\n💾 Berhasil menyimpan {total_sukses} baris fitur ke:\n   --> {file_output}")
    else:
        print("\n❌ Tidak ada fitur yang berhasil diekstrak sama sekali. CSV tidak dibuat.")

    # 3. Laporan Hasil Ekstraksi
    print("\n" + "=" * 65)
    print("📊 LAPORAN HASIL EKSTRAKSI PER HURUF")
    print("=" * 65)
    print(f"{'Huruf':<8} | {'Total':<8} | {'Sukses':<8} | {'Gagal':<8} | {'% Gagal':<8}")
    print("-" * 65)

    semua_huruf = sorted(total_per_huruf.keys())
    huruf_paling_sedikit = None
    min_sampel = float("inf")

    for h in semua_huruf:
        tot = total_per_huruf[h]
        suk = sukses_per_huruf[h]
        gag = gagal_per_huruf[h]
        persen_gag = (gag / tot * 100) if tot > 0 else 0.0

        if suk < min_sampel:
            min_sampel = suk
            huruf_paling_sedikit = h

        print(f"{h:<8} | {tot:<8} | {suk:<8} | {gag:<8} | {persen_gag:>6.1f}%")

    print("-" * 65)
    print(f"Total Gambar       : {total_gambar}")
    print(f"Berhasil Diekstrak : {total_sukses} ({(total_sukses/total_gambar*100):.1f}%)")
    print(f"Gagal Terdeteksi   : {total_gagal} ({(total_gagal/total_gambar*100):.1f}%)")
    print(f"Waktu Eksekusi     : {durasi:.1f} detik ({durasi/total_gambar*1000:.1f} ms/gambar)")
    print(f"Huruf Tersedikit   : {huruf_paling_sedikit} ({min_sampel} sampel sukses)")
    print("=" * 65)

    # 4. Peringatan person_id jika ada 'unknown'
    if ada_person_id_unknown:
        print("\n⚠️  PERINGATAN EVALUASI (person_id = 'unknown'):")
        print("   Sebagian atau semua sampel tidak memiliki ID penutur yang terdeteksi.")
        print("   Jika dievaluasi tanpa pemisahan penutur (speaker-independent), akurasi")
        print("   bisa terlihat sangat tinggi padahal model hanya menghafal penutur yang sama.")
        print("   Saran: Beri nama file seperti 'P01_A_01.jpg' atau tambahkan rekaman penutur")
        print("   berbeda lewat 'python src/collect_data.py' (Paket T3).")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SignDesk — Ekstraksi Landmark Tangan dari Folder Gambar"
    )
    parser.add_argument(
        "--input-dir",
        default=config.RAW_IMAGES_DIR,
        help=f"Folder dataset gambar (default: {config.RAW_IMAGES_DIR})"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(config.LANDMARKS_DIR, "landmarks_raw_images.csv"),
        help="Path file CSV hasil (default: data/landmarks/landmarks_raw_images.csv)"
    )
    parser.add_argument(
        "--source",
        default="dataset_gambar",
        help="Label sumber data untuk kolom 'source' (misal: 'kaggle', 'mendeley')"
    )
    parser.add_argument(
        "--person-id",
        default=None,
        help="Paksa nilai person_id untuk semua gambar yang diproses (opsional)"
    )

    args = parser.parse_args()

    ekstrak_dataset(
        folder_input=args.input_dir,
        file_output=args.output,
        sumber_data=args.source,
        person_id_override=args.person_id
    )


if __name__ == "__main__":
    main()

