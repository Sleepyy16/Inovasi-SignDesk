"""
SignDesk - src/collect_data.py
==============================
Skrip webcam interaktif untuk merekam data landmark tangan sendiri.

FITUR UTAMA (HANDS-FREE AUTO-RECORD):
-------------------------------------
- Deteksi stabilitas pose otomatis: Begitu tangan masuk ke bingkai dan diam
  selama >= 1 detik, sistem langsung otomatis merekam tanpa perlu menekan tombol!
- Dilengkapi countdown visual di layar dan audio beep lembut (feedback suara).
- Tombol SPASI tetap tersedia sebagai opsi manual.

KEBIJAKAN PRIVASI & ETIKA:
-------------------------
- Video kamera diproses langsung secara lokal di memori (RAM) dan TIDAK PERNAH
  disimpan sebagai file video atau gambar ke disk.
- Hanya koordinat angka landmark (127 fitur numerik) + label huruf + person_id
  yang dicatat ke file CSV.

KONTROL TOMBOL:
---------------
- M            : Ganti mode (AUTO HANDS-FREE vs MANUAL SPASI)
- Panah / [ ]  : Pindah ke huruf sebelumnya / berikutnya
- A - Z        : Langsung pilih huruf target
- SPASI        : Rekam manual (jika dalam mode manual)
- Q            : Simpan data ke CSV lalu keluar
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
        return 1.0  # Anggap bergerak jika salah satu tidak ada
    # Bandingkan koordinat tangan (indeks 0 s.d. 125, abaikan fitur jumlah tangan ke-126)
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
    pesan_petunjuk: str
):
    """
    Menggambar panel status interaktif (HUD) di frame webcam.
    """
    h, w = frame.shape[:2]

    # Panel atas semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 135), (20, 20, 20), -1)
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

    # Bar Countdown Stabilitas (hanya di mode otomatis)
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

    # 4. Baris 4: Pesan Petunjuk Dinamis
    cv2.putText(frame, pesan_petunjuk, (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 255, 255), 1, cv2.LINE_AA)

    # 5. Indikator Perekaman Merah Aktif
    if sedang_merekam:
        cv2.circle(frame, (w - 30, 85), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REKAM", (w - 110, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # 6. Panel Bawah: Panduan Tombol
    teks_kontrol = "Panah/[ ]: Ganti Huruf | A-Z: Pilih | M: Toggle Auto | Q: Simpan"
    cv2.putText(frame, teks_kontrol, (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def rekam_data_webcam():
    """
    Fungsi utama pengumpulan data dengan dukungan Hands-Free Auto-Record.
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
    buffer_rekaman = []

    # Variabel Kontrol Auto-Record
    mode_otomatis = True  # Default: Hands-free aktif!
    fitur_sebelumnya = None
    waktu_mulai_stabil = None
    jumlah_burst_saat_ini = 0
    waktu_terakhir_burst = 0.0
    waktu_terakhir_rekam = 0.0
    wajib_gerak_dulu = False  # Mencegah over-recording pose yang sama tanpa jeda

    durasi_indikator_rekam = 0.25

    print("\n✅ Mode Hands-Free Aktif!")
    print("-----------------------------------------------------------------")
    print("💡 CARA PAKAI:")
    print("   1. Bentuk pose huruf isyarat di depan kamera.")
    print("   2. TAHAN DIAM selama 1 detik -> Sistem otomatis merekam (ada bunyi beep)!")
    print("   3. Setelah bunyi beep, ubah sudut/jarak tangan sedikit untuk pose berikutnya.")
    print("   4. Tekan Panah Kanan atau ketik huruf untuk berpindah huruf.")
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

            if fitur is None:
                status_tangan = "Tangan Tidak Terdeteksi"
                waktu_mulai_stabil = None
                fitur_sebelumnya = None
                wajib_gerak_dulu = False
                pesan_petunjuk = "Posisikan 1 atau 2 tangan di dalam frame kamera"
            else:
                jumlah_tangan = int(fitur[-1])
                status_tangan = f"{jumlah_tangan} Tangan Terdeteksi"

                # Hitung pergeseran tangan
                pergeseran = hitung_pergeseran_pose(fitur, fitur_sebelumnya)
                fitur_sebelumnya = fitur.copy()

                if pergeseran > AMBANG_GERAKAN_STABIL:
                    # Tangan sedang bergerak
                    waktu_mulai_stabil = None
                    wajib_gerak_dulu = False
                    pesan_petunjuk = "Gerakan terdeteksi... Tahan diam posisi tangan"
                else:
                    # Tangan relatif diam
                    if wajib_gerak_dulu:
                        pesan_petunjuk = "✓ Sampel tersimpan! Ubah sudut/posisi tangan sedikit untuk sampel baru"
                    else:
                        if waktu_mulai_stabil is None:
                            waktu_mulai_stabil = waktu_sekarang

                        durasi_stabil = waktu_sekarang - waktu_mulai_stabil
                        progres_stabil = durasi_stabil / DURASI_STABIL_DETIK

                        if durasi_stabil < DURASI_STABIL_DETIK:
                            pesan_petunjuk = f"Tahan posisi... ({int(progres_stabil * 100)}%)"
                        else:
                            # --- TRIGGER OTOMATIS: Pose stabil >= 1.0 detik ---
                            pesan_petunjuk = "📸 Merekam pose stabil!"
                            if mode_otomatis and (waktu_sekarang - waktu_terakhir_burst >= config.INTERVAL_REKAM_DETIK):
                                baris = [huruf_aktif, person_id, "webcam"] + fitur.tolist()
                                buffer_rekaman.append(baris)
                                sampel_sesi[huruf_aktif] += 1
                                jumlah_burst_saat_ini += 1
                                waktu_terakhir_burst = waktu_sekarang
                                waktu_terakhir_rekam = waktu_sekarang

                                # Berikan feedback suara beep lembut
                                if jumlah_burst_saat_ini == 1:
                                    bunyikan_beep(1400, 80)

                                # Batasi per satu pose stabil maksimal MAX_BURST_PER_POSE sampel
                                if jumlah_burst_saat_ini >= MAX_BURST_PER_POSE:
                                    bunyikan_beep(900, 100)
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
                pesan_petunjuk=pesan_petunjuk
            )

            cv2.imshow("SignDesk - Perekaman Hands-Free (Tekan Q untuk Simpan)", frame)

            # Baca input keyboard
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF if key_raw != -1 else -1

            # Keluar
            if key in (ord('q'), ord('Q'), 27):
                print("\n⏹️ Menyimpan data dan menutup kamera...")
                break

            # Toggle Mode (Auto vs Manual)
            elif key in (ord('m'), ord('M')):
                mode_otomatis = not mode_otomatis
                waktu_mulai_stabil = None
                print(f"🔄 Mode diganti ke: {'AUTO HANDS-FREE' if mode_otomatis else 'MANUAL SPASI'}")

            # Rekam manual dengan SPASI (backup jika ingin manual)
            elif key == ord(' '):
                if waktu_sekarang - waktu_terakhir_rekam >= config.INTERVAL_REKAM_DETIK:
                    if fitur is not None:
                        baris = [huruf_aktif, person_id, "webcam"] + fitur.tolist()
                        buffer_rekaman.append(baris)
                        sampel_sesi[huruf_aktif] += 1
                        waktu_terakhir_rekam = waktu_sekarang
                        bunyikan_beep(1200, 60)

            # Navigasi huruf berikutnya: Panah Kanan / Down / ']' / '.'
            elif key_raw in (2555904, 2621440, 65363, 65364) or key in (ord(']'), ord('.'), ord('>')):
                idx_huruf = (idx_huruf + 1) % len(daftar_huruf)
                waktu_mulai_stabil = None
                jumlah_burst_saat_ini = 0
                wajib_gerak_dulu = False

            # Navigasi huruf sebelumnya: Panah Kiri / Up / '[' / ','
            elif key_raw in (2424832, 2490368, 65361, 65362) or key in (ord('['), ord(','), ord('<')):
                idx_huruf = (idx_huruf - 1) % len(daftar_huruf)
                waktu_mulai_stabil = None
                jumlah_burst_saat_ini = 0
                wajib_gerak_dulu = False

            # Langsung lompat ke huruf tertentu (Ketik A s.d. Z)
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

    # Simpan hasil sesi ini ke file CSV (mode APPEND)
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
            tot_lama = total_per_huruf[h]
            tot_baru = sampel_sesi[h]
            print(f"  - Huruf {h} : +{tot_baru} sampel (Total keseluruhan: {tot_lama + tot_baru})")
        print("=" * 65)
    else:
        print("\nℹ️ Tidak ada sampel yang direkam pada sesi ini. File CSV tidak berubah.")


if __name__ == "__main__":
    rekam_data_webcam()
