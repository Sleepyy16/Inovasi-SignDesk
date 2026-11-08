"""
SignDesk - src/collect_data.py
==============================
Skrip webcam interaktif untuk merekam data landmark tangan sendiri.

FITUR UTAMA:
------------
- HANDS-FREE AUTO-RECORD: Tahan posisi tangan diam selama >= 1 detik,
  sistem otomatis merekam tanpa perlu menekan tombol apa pun!
- UNDO & ANTI-SALAH:
  * BACKSPACE / U : Batalkan rekaman pose terakhir jika salah pose
  * C             : Hapus semua rekaman huruf aktif pada sesi ini
  * ESC           : KELUAR TANPA MENYIMPAN (buang semua rekaman sesi ini)
  * Q             : SIMPAN data sesi ini ke CSV lalu keluar

KEBIJAKAN PRIVASI & ETIKA:
-------------------------
- Video kamera diproses langsung secara lokal di memori (RAM) dan TIDAK PERNAH
  disimpan sebagai file video atau gambar ke disk.
- Hanya koordinat angka landmark (127 fitur numerik) + label huruf + person_id
  yang dicatat ke file CSV.
"""

import sys
import os
import time
from collections import defaultdict

import cv2
import pandas as pd
import numpy as np

# Audio feedback untuk Windows (opsional, aman jika gagal)
try:
    import winsound
    def bunyikan_beep(frekuensi=1200, durasi_ms=80):
        try:
            winsound.Beep(frekuensi, durasi_ms)
        except Exception:
            pass
except ImportError:
    def bunyikan_beep(*args, **kwargs):
        pass

# Tambahkan root proyek ke sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import config
from src.features import buat_detektor, ekstrak_fitur, gambar_overlay, nama_kolom_fitur

# Pengaturan Hands-Free Auto-Record
DURASI_STABIL_DETIK = 1.0       # Tahan posisi diam selama 1.0 detik
AMBANG_GERAKAN_STABIL = 0.022   # Toleransi pergeseran koordinat (landmark jitter)
MAX_BURST_PER_POSE = 8          # Maksimal sampel terekam per satu kali pose diam
COOLDOWN_BURST_DETIK = 0.6      # Jeda setelah 1 rentetan rekam selesai


def muat_statistik_lama(csv_path: str) -> tuple[dict, set]:
    """
    Membaca CSV yang sudah ada untuk menghitung jumlah sampel per huruf.
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
            pass

    return total_per_huruf, penutur_tercatat


def tampilkan_persetujuan_dan_minta_id() -> str | None:
    """
    Menampilkan persetujuan privasi dan meminta ID penutur.
    """
    print("\n" + "=" * 65)
    print("  SignDesk — Pengumpulan Data Hands-Free via Webcam")
    print("=" * 65)
    print("PERNYATAAN PRIVASI:")
    print("1. Kamera HANYA memproses koordinat tangan secara real-time.")
    print("2. Gambar wajah dan rekaman video TIDAK DISIMPAN ke disk.")
    print("3. Hanya koordinat angka landmark + label + person_id yang disimpan.")
    print("=" * 65)

    konfirmasi = input("Apakah Anda menyetujui ketentuan di atas? (y/n): ").strip().lower()
    if konfirmasi not in ("y", "ya", "yes"):
        print("❌ Perekaman dibatalkan.")
        return None

    while True:
        pid = input("\nMasukkan ID Penutur (contoh: P01, P02, NAMA): ").strip().upper()
        if pid:
            pid_bersih = "".join(c for c in pid if c.isalnum() or c in ("_", "-"))
            if pid_bersih:
                return pid_bersih
        print("⚠️ ID Penutur tidak boleh kosong.")


def hitung_pergeseran_pose(fitur_sekarang: np.ndarray, fitur_sebelumnya: np.ndarray) -> float:
    """
    Menghitung rata-rata selisih koordinat landmark antar frame untuk mendeteksi
    apakah tangan sedang bergerak atau diam (stabil).
    """
    if fitur_sekarang is None or fitur_sebelumnya is None:
        return 1.0
    diff = np.mean(np.abs(fitur_sekarang[:126] - fitur_sebelumnya[:126]))
    return float(diff)


def gambar_hud(
    frame: np.ndarray,
    huruf_aktif: str,
    person_id: str,
    total_lama: int,
    total_sesi: int,
    status_tangan: str,
    sedang_merekam: bool,
    mode_otomatis: bool,
    progres_stabil: float,
    pesan_petunjuk: str,
    pesan_notifikasi: str = ""
):
    """
    Menggambar panel status interaktif (HUD) di frame webcam.
    """
    h, w = frame.shape[:2]

    # Panel atas semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (20, 20, 20), -1)
    # Panel bawah
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # 1. Baris 1: Huruf target & Mode rekam
    is_eksp = huruf_aktif in config.HURUF_EKSPERIMENTAL
    label_huruf = f"HURUF: [ {huruf_aktif} ]"
    if is_eksp:
        label_huruf += " (Eksperimental)"
    cv2.putText(frame, label_huruf, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

    teks_mode = "MODE: [AUTO HANDS-FREE]" if mode_otomatis else "MODE: [MANUAL SPASI]"
    warna_mode = (0, 255, 128) if mode_otomatis else (200, 200, 200)
    cv2.putText(frame, teks_mode, (w - 290, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_mode, 2, cv2.LINE_AA)

    # 2. Baris 2: Statistik Sampel & Penutur
    info_sampel = f"Sampel: {total_lama + total_sesi} (Sesi ini: +{total_sesi}) | Penutur: {person_id}"
    cv2.putText(frame, info_sampel, (15, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Baris 3: Status Tangan & Countdown Kestabilan Pose
    warna_status = (0, 255, 0) if "Terdeteksi" in status_tangan else (0, 165, 255)
    cv2.putText(frame, status_tangan, (15, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna_status, 1, cv2.LINE_AA)

    if mode_otomatis:
        bar_x, bar_y, bar_w, bar_h = 240, 80, 200, 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        terisi = int(bar_w * min(progres_stabil, 1.0))
        warna_bar = (0, 255, 0) if progres_stabil >= 1.0 else (0, 215, 255)
        if terisi > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + terisi, bar_y + bar_h), warna_bar, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        teks_tahan = "Tahan Posisi..." if progres_stabil < 1.0 else "MEREKAM!"
        cv2.putText(frame, teks_tahan, (bar_x + bar_w + 10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_bar, 1, cv2.LINE_AA)

    # 4. Baris 4: Pesan Notifikasi (Undo/Hapus) atau Petunjuk Dinamis
    if pesan_notifikasi:
        cv2.putText(frame, pesan_notifikasi, (15, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 150, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, pesan_petunjuk, (15, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 255, 255), 1, cv2.LINE_AA)

    # 5. Indikator Perekaman Merah Aktif
    if sedang_merekam:
        cv2.circle(frame, (w - 30, 85), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REKAM", (w - 110, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # 6. Panel Bawah: Panduan Tombol Lengkap (Termasuk Undo & Batal)
    teks_kontrol = "Panah/A-Z: Huruf | Backspace/U: Undo | C: Reset Huruf | Q: Simpan | ESC: Batal"
    cv2.putText(frame, teks_kontrol, (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)


def rekam_data_webcam():
    """
    Fungsi utama pengumpulan data dengan dukungan Hands-Free Auto-Record,
    fitur Undo, dan pembatalan penyimpanan.
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
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, config.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TINGGI_KAMERA)

    try:
        detektor = buat_detektor(mode_gambar_statis=False)
    except Exception as e:
        print(f"❌ Gagal membuat detektor MediaPipe: {e}")
        kamera.release()
        return

    daftar_huruf = config.HURUF_DIDUKUNG
    idx_huruf = 0
    sampel_sesi = defaultdict(int)
    buffer_rekaman = []  # Menyimpan baris data sesi ini

    # Riwayat batch perekaman untuk fungsi Undo: list of (huruf, jumlah_sampel)
    riwayat_batch = []

    # Kontrol Notifikasi Layar
    teks_notifikasi = ""
    waktu_notifikasi = 0.0

    # Variabel Kontrol Auto-Record
    mode_otomatis = True
    fitur_sebelumnya = None
    waktu_mulai_stabil = None
    jumlah_burst_saat_ini = 0
    waktu_terakhir_burst = 0.0
    waktu_terakhir_rekam = 0.0
    wajib_gerak_dulu = False

    durasi_indikator_rekam = 0.25
    simpan_saat_keluar = False  # Hanya True jika user menekan 'Q'

    print("\n✅ Kamera aktif dalam mode Hands-Free Auto-Record!")
    print("-----------------------------------------------------------------")
    print("💡 PANDUAN PENGONTROLAN:")
    print("   - Tahan tangan diam 1 detik -> Otomatis rekam 8 sampel (ada bunyi beep)")
    print("   - Tombol BACKSPACE atau 'U' -> Batalkan/Undo pose terakhir")
    print("   - Tombol 'C'                -> Hapus semua rekaman huruf saat ini di sesi ini")
    print("   - Tombol 'Q'                -> SIMPAN ke CSV dan selesai")
    print("   - Tombol 'ESC'              -> KELUAR TANPA SIMPAN (buang semua)")
    print("-----------------------------------------------------------------\n")

    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                break

            frame = cv2.flip(frame, 1)
            ts_ms = int(time.time() * 1000)
            waktu_sekarang = time.time()

            # Ekstrak fitur
            fitur, hasil = ekstrak_fitur(frame, detektor, timestamp_ms=ts_ms)
            frame = gambar_overlay(frame, hasil)

            huruf_aktif = daftar_huruf[idx_huruf]
            sedang_merekam = (waktu_sekarang - waktu_terakhir_rekam) < durasi_indikator_rekam
            progres_stabil = 0.0
            pesan_petunjuk = "Bentuk pose tangan di depan kamera..."

            # Periksa apakah notifikasi banner masih aktif (tampil selama 2 detik)
            pesan_notifikasi_aktif = teks_notifikasi if (waktu_sekarang - waktu_notifikasi < 2.0) else ""

            if fitur is None:
                status_tangan = "Tangan Tidak Terdeteksi"
                waktu_mulai_stabil = None
                fitur_sebelumnya = None
                wajib_gerak_dulu = False
                pesan_petunjuk = "Posisikan 1 atau 2 tangan di dalam frame kamera"
            else:
                jumlah_tangan = int(fitur[-1])
                status_tangan = f"{jumlah_tangan} Tangan Terdeteksi"

                pergeseran = hitung_pergeseran_pose(fitur, fitur_sebelumnya)
                fitur_sebelumnya = fitur.copy()

                if pergeseran > AMBANG_GERAKAN_STABIL:
                    waktu_mulai_stabil = None
                    wajib_gerak_dulu = False
                    pesan_petunjuk = "Gerakan terdeteksi... Tahan diam posisi tangan"
                else:
                    if wajib_gerak_dulu:
                        pesan_petunjuk = "✓ Tersimpan! Ubah posisi/sudut tangan sedikit untuk pose baru"
                    else:
                        if waktu_mulai_stabil is None:
                            waktu_mulai_stabil = waktu_sekarang

                        durasi_stabil = waktu_sekarang - waktu_mulai_stabil
                        progres_stabil = durasi_stabil / DURASI_STABIL_DETIK

                        if durasi_stabil < DURASI_STABIL_DETIK:
                            pesan_petunjuk = f"Tahan posisi... ({int(progres_stabil * 100)}%)"
                        else:
                            # TRIGGER AUTO-RECORD
                            pesan_petunjuk = "📸 Merekam pose stabil!"
                            if mode_otomatis and (waktu_sekarang - waktu_terakhir_burst >= config.INTERVAL_REKAM_DETIK):
                                baris = [huruf_aktif, person_id, "webcam"] + fitur.tolist()
                                buffer_rekaman.append(baris)
                                sampel_sesi[huruf_aktif] += 1
                                jumlah_burst_saat_ini += 1
                                waktu_terakhir_burst = waktu_sekarang
                                waktu_terakhir_rekam = waktu_sekarang

                                if jumlah_burst_saat_ini == 1:
                                    bunyikan_beep(1400, 80)

                                if jumlah_burst_saat_ini >= MAX_BURST_PER_POSE:
                                    bunyikan_beep(900, 100)
                                    # Catat riwayat batch untuk fitur Undo
                                    riwayat_batch.append((huruf_aktif, jumlah_burst_saat_ini))
                                    wajib_gerak_dulu = True
                                    jumlah_burst_saat_ini = 0
                                    waktu_mulai_stabil = None

            # Gambar HUD interaktif
            gambar_hud(
                frame=frame,
                huruf_aktif=huruf_aktif,
                person_id=person_id,
                total_lama=total_per_huruf[huruf_aktif],
                total_sesi=sampel_sesi[huruf_aktif],
                status_tangan=status_tangan,
                sedang_merekam=sedang_merekam,
                mode_otomatis=mode_otomatis,
                progres_stabil=progres_stabil,
                pesan_petunjuk=pesan_petunjuk,
                pesan_notifikasi=pesan_notifikasi_aktif
            )

            cv2.imshow("SignDesk - Perekaman Hands-Free (Q: Simpan | ESC: Batal)", frame)

            # Baca input keyboard
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF if key_raw != -1 else -1

            # 1. SIMPAN DAN KELUAR (Q)
            if key in (ord('q'), ord('Q')):
                print("\n💾 Menekan 'Q': Menyimpan data sesi ini ke CSV...")
                simpan_saat_keluar = True
                break

            # 2. KELUAR TANPA SIMPAN (ESC)
            elif key == 27:
                print("\n🚫 Menekan 'ESC': KELUAR TANPA MENYIMPAN. Semua data sesi ini dibuang!")
                simpan_saat_keluar = False
                break

            # 3. UNDO POSE TERAKHIR (BACKSPACE atau U)
            elif key in (8, ord('u'), ord('U')):
                if riwayat_batch:
                    huruf_batal, jml_batal = riwayat_batch.pop()
                    # Buang jml_batal baris terakhir dari buffer
                    buffer_rekaman = buffer_rekaman[:-jml_batal]
                    sampel_sesi[huruf_batal] = max(0, sampel_sesi[huruf_batal] - jml_batal)
                    teks_notifikasi = f"↩️ UNDO: {jml_batal} sampel huruf {huruf_batal} dibatalkan!"
                    waktu_notifikasi = waktu_sekarang
                    bunyikan_beep(500, 150)
                    print(f"\n↩️ {teks_notifikasi}")
                elif buffer_rekaman:
                    # Jika ada sisa rekaman single-shot (manual)
                    baris_terakhir = buffer_rekaman.pop()
                    huruf_batal = baris_terakhir[0]
                    sampel_sesi[huruf_batal] = max(0, sampel_sesi[huruf_batal] - 1)
                    teks_notifikasi = f"↩️ UNDO: 1 sampel huruf {huruf_batal} dibatalkan!"
                    waktu_notifikasi = waktu_sekarang
                    bunyikan_beep(500, 150)
                else:
                    teks_notifikasi = "⚠️ Belum ada sampel yang bisa di-Undo."
                    waktu_notifikasi = waktu_sekarang

            # 4. KOSONGKAN REKAMAN HURUF INI DI SESI INI (Tombol 'C')
            elif key in (ord('c'), ord('C')):
                jml_hapus = sampel_sesi[huruf_aktif]
                if jml_hapus > 0:
                    # Hapus semua baris huruf_aktif dari buffer
                    buffer_rekaman = [b for b in buffer_rekaman if b[0] != huruf_aktif]
                    riwayat_batch = [r for r in riwayat_batch if r[0] != huruf_aktif]
                    sampel_sesi[huruf_aktif] = 0
                    teks_notifikasi = f"🗑️ Bersih: {jml_hapus} sampel sesi ini untuk '{huruf_aktif}' dihapus!"
                    waktu_notifikasi = waktu_sekarang
                    bunyikan_beep(400, 200)
                    print(f"\n🗑️ {teks_notifikasi}")
                else:
                    teks_notifikasi = f"ℹ️ Huruf '{huruf_aktif}' belum punya sampel di sesi ini."
                    waktu_notifikasi = waktu_sekarang

            # 5. TOGGLE MODE (Auto vs Manual)
            elif key in (ord('m'), ord('M')):
                mode_otomatis = not mode_otomatis
                waktu_mulai_stabil = None
                teks_notifikasi = f"Mode: {'AUTO HANDS-FREE' if mode_otomatis else 'MANUAL SPASI'}"
                waktu_notifikasi = waktu_sekarang

            # 6. REKAM MANUAL (SPASI)
            elif key == ord(' '):
                if waktu_sekarang - waktu_terakhir_rekam >= config.INTERVAL_REKAM_DETIK:
                    if fitur is not None:
                        baris = [huruf_aktif, person_id, "webcam"] + fitur.tolist()
                        buffer_rekaman.append(baris)
                        sampel_sesi[huruf_aktif] += 1
                        waktu_terakhir_rekam = waktu_sekarang
                        bunyikan_beep(1200, 60)

            # 7. NAVIGASI HURUF
            elif key_raw in (2555904, 2621440, 65363, 65364) or key in (ord(']'), ord('.'), ord('>')):
                idx_huruf = (idx_huruf + 1) % len(daftar_huruf)
                waktu_mulai_stabil = None
                jumlah_burst_saat_ini = 0
                wajib_gerak_dulu = False

            elif key_raw in (2424832, 2490368, 65361, 65362) or key in (ord('['), ord(','), ord('<')):
                idx_huruf = (idx_huruf - 1) % len(daftar_huruf)
                waktu_mulai_stabil = None
                jumlah_burst_saat_ini = 0
                wajib_gerak_dulu = False

            elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
                huruf_input = chr(key).upper()
                if huruf_input in daftar_huruf:
                    idx_huruf = daftar_huruf.index(huruf_input)
                    waktu_mulai_stabil = None
                    jumlah_burst_saat_ini = 0
                    wajib_gerak_dulu = False

    finally:
        kamera.release()
        detektor.close()
        cv2.destroyAllWindows()

    # Eksekusi Penyimpanan
    if simpan_saat_keluar:
        total_didapat = len(buffer_rekaman)
        if total_didapat > 0:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            file_baru = not os.path.exists(csv_path)

            kolom = ["label", "person_id", "source"] + nama_kolom_fitur()
            df_baru = pd.DataFrame(buffer_rekaman, columns=kolom)

            df_baru.to_csv(csv_path, mode="a", header=file_baru, index=False)

            print("\n" + "=" * 65)
            print("💾 RINGKASAN DATA BERHASIL DISIMPAN (MODE APPEND)")
            print("=" * 65)
            print(f"File Target  : {csv_path}")
            print(f"Penutur      : {person_id}")
            print(f"Total Sampel : +{total_didapat} baris fitur baru ditambahkan\n")
            print("Rincian per huruf sesi ini:")
            for h in sorted(sampel_sesi.keys()):
                if sampel_sesi[h] > 0:
                    tot_lama = total_per_huruf[h]
                    tot_baru = sampel_sesi[h]
                    print(f"  - Huruf {h} : +{tot_baru} sampel (Total keseluruhan: {tot_lama + tot_baru})")
            print("=" * 65)
        else:
            print("\nℹ️ Tidak ada sampel yang terekam pada sesi ini. File CSV tidak berubah.")
    else:
        print("\n" + "=" * 65)
        print("🚫 PEMBATALAN: Data sesi ini TIDAK DISIMPAN ke CSV.")
        print("   File CSV database lama tetap aman dan tidak mengalami perubahan.")
        print("=" * 65)


if __name__ == "__main__":
    rekam_data_webcam()
