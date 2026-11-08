"""
SignDesk - src/collect_data.py
==============================
Skrip webcam interaktif untuk merekam data landmark tangan sendiri.

KEBIJAKAN PRIVASI & ETIKA:
-------------------------
- Video kamera diproses langsung secara lokal di memori (RAM) dan TIDAK PERNAH
  disimpan sebagai file video atau gambar ke disk.
- Hanya koordinat angka landmark (127 fitur numerik) + label huruf + person_id
  yang dicatat ke file CSV.
- Sebelum perekaman dimulai, pengguna wajib memberikan persetujuan.

PANDUAN VARIASI DATA (AGAR MODEL TAHAN UJI):
--------------------------------------------
Saat merekam tiap huruf:
1. Variasikan jarak tangan dari webcam (dekat, sedang, agak jauh).
2. Variasikan sudut tangan (sedikit miring ke kiri/kanan/atas/bawah).
3. Variasikan pencahayaan (terang, redup, dari samping).
4. Ubah posisi tangan sedikit di tiap rekaman (jangan diam mematung).

KONTROL TOMBOL:
---------------
- SPASI        : Rekam 1 sampel (atau tahan untuk merekam berturut-turut)
- A - Z        : Langsung pilih huruf target
- [ / ]        : Geser ke huruf sebelumnya / berikutnya
- Q            : Simpan data sesi ini ke CSV lalu keluar
"""

import sys
import os
import time
from collections import defaultdict

import cv2
import pandas as pd
import numpy as np

# Tambahkan root proyek ke sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import config
from src.features import buat_detektor, ekstrak_fitur, gambar_overlay, nama_kolom_fitur


def muat_statistik_lama(csv_path: str) -> tuple[dict, set]:
    """
    Membaca CSV yang sudah ada untuk menghitung jumlah sampel per huruf
    dan mendata ID penutur yang sudah pernah tercatat.
    """
    total_per_huruf = defaultdict(int)
    penutur_tercatat = set()

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, usecols=["label", "person_id"])
            for _, row in df.iterrows():
                total_per_huruf[str(row["label"]).upper()] += 1
                penutur_tercatat.add(str(row["person_id"]))
        except Exception:
            # Jika file kosong atau rusak, mulai dari nol
            pass

    return total_per_huruf, penutur_tercatat


def tampilkan_persetujuan_dan_minta_id() -> str | None:
    """
    Menampilkan lembar persetujuan etika & privasi, lalu meminta person_id.
    """
    print("\n" + "=" * 65)
    print("  SignDesk — Pengumpulan Data Landmark BISINDO via Webcam")
    print("=" * 65)
    print("PERNYATAAN PRIVASI & PERSETUJUAN:")
    print("1. Kamera HANYA memproses koordinat tangan secara real-time.")
    print("2. Gambar wajah, foto, atau rekaman video TIDAK DISIMPAN ke disk.")
    print("3. Data yang disimpan hanyalah angka landmark koordinat tangan (CSV),")
    print("   label huruf, dan ID penutur (misal: P01).")
    print("4. Data digunakan semata-mata untuk pelatihan model AI SignDesk.")
    print("=" * 65)

    konfirmasi = input("Apakah Anda menyetujui ketentuan di atas? (y/n): ").strip().lower()
    if konfirmasi not in ("y", "ya", "yes"):
        print("❌ Perekaman dibatalkan oleh pengguna.")
        return None

    # Minta ID penutur
    while True:
        pid = input("\nMasukkan ID Penutur (contoh: P01, P02, ALIF): ").strip().upper()
        if pid:
            # Pastikan tidak ada spasi atau karakter aneh
            pid_bersih = "".join(c for c in pid if c.isalnum() or c in ("_", "-"))
            if pid_bersih:
                return pid_bersih
        print("⚠️ ID Penutur tidak boleh kosong. Silakan coba lagi.")


def gambar_hud(
    frame: np.ndarray,
    huruf_aktif: str,
    person_id: str,
    total_lama: int,
    total_sesi: int,
    status_tangan: str,
    sedang_merekam: bool
):
    """
    Menggambar informasi status (HUD - Heads-Up Display) pada frame video.
    """
    h, w = frame.shape[:2]

    # 1. Overlay kotak semi-transparan di bagian atas untuk panel info
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 125), (20, 20, 20), -1)
    # Kotak bawah untuk panduan tombol
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 2. Huruf target yang sedang direkam
    is_eksp = huruf_aktif in config.HURUF_EKSPERIMENTAL
    label_huruf = f"HURUF: [ {huruf_aktif} ]"
    if is_eksp:
        label_huruf += " (Eksperimental - Gerakan)"

    cv2.putText(frame, label_huruf, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

    # 3. Statistik sampel
    info_sampel = f"Total: {total_lama + total_sesi} sampel  (Sesi ini: +{total_sesi})"
    cv2.putText(frame, info_sampel, (15, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Status penutur & deteksi tangan
    info_status = f"Penutur: {person_id} | {status_tangan}"
    warna_status = (0, 255, 0) if "Terdeteksi" in status_tangan else (0, 165, 255)
    cv2.putText(frame, info_status, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna_status, 1, cv2.LINE_AA)

    # 5. Panduan variasi di layar
    teks_panduan = "Tips: Ubah jarak, sudut, cahaya & posisi tangan saat rekam!"
    cv2.putText(frame, teks_panduan, (15, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1, cv2.LINE_AA)

    # 6. Indikator perekaman aktif
    if sedang_merekam:
        # Lingkaran merah kedip di pojok kanan atas
        cv2.circle(frame, (w - 30, 35), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REKAM", (w - 110, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # 7. Baris instruksi tombol di bagian bawah
    teks_kontrol = "SPASI: Rekam | Panah / [ ]: Ganti Huruf | A-Z: Pilih | Q: Simpan"
    cv2.putText(frame, teks_kontrol, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def rekam_data_webcam():
    """
    Fungsi utama pengumpulan data dari webcam.
    """
    person_id = tampilkan_persetujuan_dan_minta_id()
    if not person_id:
        return

    csv_path = config.LANDMARKS_WEBCAM_CSV
    total_per_huruf, _ = muat_statistik_lama(csv_path)

    print(f"\n🎥 Membuka kamera (indeks: {config.INDEKS_KAMERA})...")
    kamera = cv2.VideoCapture(config.INDEKS_KAMERA)
    if not kamera.isOpened():
        print(f"❌ Gagal membuka kamera dengan indeks {config.INDEKS_KAMERA}.")
        print("   Periksa apakah kamera sedang dipakai oleh aplikasi lain (Zoom, Teams, dll).")
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, config.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TINGGI_KAMERA)

    # Detektor MediaPipe dalam mode VIDEO untuk tracking tangan yang lancar
    try:
        detektor = buat_detektor(mode_gambar_statis=False)
    except Exception as e:
        print(f"❌ Gagal membuat detektor MediaPipe: {e}")
        kamera.release()
        return

    daftar_huruf = config.HURUF_DIDUKUNG
    idx_huruf = 0
    sampel_sesi = defaultdict(int)
    buffer_rekaman = []  # Menyimpan baris CSV yang terkumpul di sesi ini

    waktu_terakhir_rekam = 0.0
    sedang_merekam = False
    durasi_indikator_rekam = 0.2  # detik untuk menampilkan animasi indikator rekam

    print("\n✅ Kamera aktif!")
    print("-----------------------------------------------------------------")
    print("💡 TIPS VARIASI:")
    print("   - Gerakkan tangan sedikit (ubah sudut & jarak) saat menahan SPASI.")
    print("   - Coba variasikan pencahayaan ruangan bila memungkinkan.")
    print("   - Huruf dua tangan: pastikan kedua tangan masuk ke dalam bingkai kamera.")
    print("-----------------------------------------------------------------\n")

    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                print("⚠️ Gagal membaca frame dari webcam.")
                break

            # Cerminkan tampilan webcam (mirror) agar nyaman seperti bercermin
            frame = cv2.flip(frame, 1)

            # Hitung timestamp milidetik yang selalu meningkat (syarat MediaPipe Video Mode)
            ts_ms = int(time.time() * 1000)

            # Ekstrak fitur & dapatkan hasil landmark
            fitur, hasil = ekstrak_fitur(frame, detektor, timestamp_ms=ts_ms)

            # Gambar visualisasi skeleton tangan di layar
            frame = gambar_overlay(frame, hasil)

            # Tentukan status teks deteksi tangan
            if fitur is None:
                status_tangan = "Tangan Tidak Terdeteksi"
            else:
                jumlah_tangan = int(fitur[-1])
                status_tangan = f"{jumlah_tangan} Tangan Terdeteksi"

            # Periksa apakah indikator rekam masih aktif
            sedang_merekam = (time.time() - waktu_terakhir_rekam) < durasi_indikator_rekam

            # Gambar seluruh HUD / petunjuk visual
            huruf_aktif = daftar_huruf[idx_huruf]
            gambar_hud(
                frame=frame,
                huruf_aktif=huruf_aktif,
                person_id=person_id,
                total_lama=total_per_huruf[huruf_aktif],
                total_sesi=sampel_sesi[huruf_aktif],
                status_tangan=status_tangan,
                sedang_merekam=sedang_merekam
            )

            cv2.imshow("SignDesk - Pengumpulan Data Landmark (Tekan Q untuk Simpan)", frame)

            # Baca input keyboard (waitKeyEx mendukung tombol panah di Windows/Linux)
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF if key_raw != -1 else -1

            # 1. Keluar (Q / ESC)
            if key in (ord('q'), ord('Q'), 27):
                print("\n⏹️ Menghentikan sesi perekaman...")
                break

            # 2. Rekam sampel (SPASI)
            elif key == ord(' '):
                waktu_sekarang = time.time()
                # Batasi frekuensi rekam agar sampel tidak terlalu identik (throttle)
                if waktu_sekarang - waktu_terakhir_rekam >= config.INTERVAL_REKAM_DETIK:
                    if fitur is not None:
                        # Susun satu baris: [label, person_id, source, f_000, ..., f_126]
                        baris = [huruf_aktif, person_id, "webcam"] + fitur.tolist()
                        buffer_rekaman.append(baris)
                        sampel_sesi[huruf_aktif] += 1
                        waktu_terakhir_rekam = waktu_sekarang
                    else:
                        # Tangan tidak terdeteksi, beri peringatan di konsol
                        sys.stdout.write(f"\r⚠️ [Huruf {huruf_aktif}] Tangan tidak terdeteksi. Posisikan tangan di depan kamera.")
                        sys.stdout.flush()

            # 3. Navigasi huruf berikutnya: Panah Kanan / Down, atau tombol ']' / '.'
            elif key_raw in (2555904, 2621440, 65363, 65364) or key in (ord(']'), ord('.'), ord('>')):
                idx_huruf = (idx_huruf + 1) % len(daftar_huruf)

            # 4. Navigasi huruf sebelumnya: Panah Kiri / Up, atau tombol '[' / ','
            elif key_raw in (2424832, 2490368, 65361, 65362) or key in (ord('['), ord(','), ord('<')):
                idx_huruf = (idx_huruf - 1) % len(daftar_huruf)

            # 5. Langsung lompat ke huruf tertentu (Ketik A s.d. Z)
            elif ord('a') <= key <= ord('z'):
                huruf_input = chr(key).upper()
                if huruf_input in daftar_huruf:
                    idx_huruf = daftar_huruf.index(huruf_input)
            elif ord('A') <= key <= ord('Z'):
                huruf_input = chr(key)
                if huruf_input in daftar_huruf:
                    idx_huruf = daftar_huruf.index(huruf_input)

    finally:
        kamera.release()
        detektor.close()
        cv2.destroyAllWindows()

    # Simpan hasil sesi ini ke file CSV (mode APPEND)
    total_didapat = len(buffer_rekaman)
    if total_didapat > 0:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        file_baru = not os.path.exists(csv_path)

        kolom = ["label", "person_id", "source"] + nama_kolom_fitur()
        df_baru = pd.DataFrame(buffer_rekaman, columns=kolom)

        # Simpan (append jika file sudah ada)
        df_baru.to_csv(csv_path, mode="a", header=file_baru, index=False)

        print("\n" + "=" * 65)
        print("💾 RINGKASAN DATA BERHASIL DISIMPAN")
        print("=" * 65)
        print(f"File Target  : {csv_path}")
        print(f"Penutur      : {person_id}")
        print(f"Total Sampel : {total_didapat} baris fitur baru ditambahkan\n")
        print("Rincian per huruf sesi ini:")
        for h in sorted(sampel_sesi.keys()):
            tot_lama = total_per_huruf[h]
            tot_baru = sampel_sesi[h]
            print(f"  - Huruf {h} : +{tot_baru} sampel (Total keseluruhan: {tot_lama + tot_baru})")
        print("=" * 65)
    else:
        print("\nℹ️ Tidak ada sampel yang direkam pada sesi ini. File CSV tidak berubah.")


if __name__ == "__main__":
    rekam_data_webcam()

