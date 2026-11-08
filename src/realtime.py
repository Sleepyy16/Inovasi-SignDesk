"""Demo real-time penerjemah alfabet BISINDO menggunakan OpenCV.

Fitur:
- Tampilan kamera dicerminkan (mirror) agar natural bagi pengguna.
- Overlay titik landmark tangan (sampai 2 tangan).
- Menampilkan huruf prediksi saat ini, skor keyakinan, dan bar progres stabilisasi (1 detik).
- Menampilkan teks hasil ejaan di bagian bawah layar.
- Kontrol keyboard:
    SPASI     : tambah spasi
    BACKSPACE : hapus 1 huruf terakhir
    R         : reset / kosongkan teks
    Q / ESC   : keluar

PRIVASI: video hanya diproses di memori dan TIDAK disimpan ke disk.

Jalankan dari folder utama:
    python src/realtime.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import joblib
import numpy as np

import config
import features
from stabilizer import PenstabilPrediksi

TEKS_DISCLAIMER = "Alat bantu komunikasi dasar, bukan pengganti juru bahasa isyarat."


def muat_model():
    """Muat model terlatih dari models/model_terbaik.joblib."""
    if not config.MODEL_PATH.exists():
        print(f"Model belum ditemukan di: {config.MODEL_PATH}")
        print("Jalankan dulu pelatihan model:  python src/train.py")
        return None, None
    try:
        paket = joblib.load(config.MODEL_PATH)
        return paket["model"], paket["nama_model"]
    except Exception as error:
        print(f"Gagal memuat model ({error}). Coba jalankan ulang: python src/train.py")
        return None, None


def prediksi_satu_frame(model, vektor):
    """Kembalikan (huruf_terbaik, skor_keyakinan) dari vektor fitur 127 angka."""
    if vektor is None:
        return None, 0.0
    X = np.asarray(vektor, dtype=np.float32).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        peluang = model.predict_proba(X)[0]
        indeks = int(np.argmax(peluang))
        return str(model.classes_[indeks]), float(peluang[indeks])
    prediksi = str(model.predict(X)[0])
    return prediksi, 1.0


def gambar_landmark(frame, daftar_tangan):
    """Gambar titik-titik landmark tangan di atas frame."""
    tinggi, lebar = frame.shape[:2]
    for tangan in daftar_tangan:
        for x, y, _ in tangan:
            cv2.circle(frame, (int(x * lebar), int(y * tinggi)), 4, (0, 255, 0), -1)


def gambar_antarmuka(frame, huruf_mentah, keyakinan, status_stabil, nama_model):
    """Gambar panel atas (prediksi & bar stabil) serta panel bawah (teks & kontrol)."""
    tinggi, lebar = frame.shape[:2]

    # --- Panel Atas: latar gelap semi-transparan ---
    lapisan = frame.copy()
    cv2.rectangle(lapisan, (0, 0), (lebar, 95), (20, 20, 20), -1)
    cv2.rectangle(lapisan, (0, tinggi - 105), (lebar, tinggi), (20, 20, 20), -1)
    cv2.addWeighted(lapisan, 0.75, frame, 0.25, 0, frame)

    huruf_tampil = status_stabil["huruf_stabil"] or (huruf_mentah if huruf_mentah else "-")
    persen = int(round(keyakinan * 100))
    warna_huruf = (0, 255, 0) if status_stabil["huruf_stabil"] else (0, 200, 255)

    teks_prediksi = f"Huruf: {huruf_tampil}  ({persen}%)"
    if huruf_tampil in config.HURUF_EKSPERIMENTAL:
        teks_prediksi += " [eksperimental]"
    cv2.putText(frame, teks_prediksi, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, warna_huruf, 2)

    cv2.putText(frame, f"Model: {nama_model}", (lebar - 210, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Bar progres stabilisasi (menuju 1 detik)
    lebar_bar_maks = 260
    progres = float(status_stabil["progres"])
    lebar_isi = int(lebar_bar_maks * progres)
    cv2.rectangle(frame, (15, 52), (15 + lebar_bar_maks, 72), (100, 100, 100), 1)
    if lebar_isi > 0:
        cv2.rectangle(frame, (15, 52), (15 + lebar_isi, 72), (0, 255, 120), -1)
    cv2.putText(frame, "Tahan ~1 dtk untuk ketik", (290, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # --- Panel Bawah: teks hasil ejaan & petunjuk tombol ---
    teks_ejaan = status_stabil["teks"]
    # Tampilkan maksimal 32 karakter terakhir agar muat di layar
    potongan_teks = teks_ejaan[-32:] if len(teks_ejaan) > 32 else teks_ejaan
    cv2.putText(frame, f"Teks: {potongan_teks}_", (15, tinggi - 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(
        frame,
        "SPASI: spasi | BACKSPACE: hapus | R: reset | Q: keluar",
        (15, tinggi - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 220, 255),
        1,
    )
    cv2.putText(
        frame,
        TEKS_DISCLAIMER,
        (15, tinggi - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (180, 180, 180),
        1,
    )


def main():
    model, nama_model = muat_model()
    if model is None:
        return

    kamera = cv2.VideoCapture(config.INDEKS_KAMERA)
    if not kamera.isOpened():
        print(f"Kamera indeks {config.INDEKS_KAMERA} tidak bisa dibuka.")
        print("Pastikan kamera tidak dipakai aplikasi lain, atau ubah INDEKS_KAMERA di config.py.")
        return
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, config.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TINGGI_KAMERA)

    detektor = features.buat_detektor(gambar_statis=False)
    penstabil = PenstabilPrediksi()

    print("Demo real-time SignDesk berjalan.")
    print("Kontrol di jendela video: SPASI=spasi, BACKSPACE=hapus, R=reset, Q=keluar.")
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
                frame = cv2.flip(frame, 1)

            huruf_mentah, keyakinan = prediksi_satu_frame(model, vektor)
            status_stabil = penstabil.perbarui(huruf_mentah, keyakinan)

            gambar_antarmuka(frame, huruf_mentah, keyakinan, status_stabil, nama_model)
            cv2.imshow("SignDesk - Demo Real-Time BISINDO", frame)

            tombol = cv2.waitKey(1) & 0xFF
            if tombol in (ord("q"), ord("Q"), 27):  # Q atau ESC
                break
            elif tombol == 32:                      # SPASI
                penstabil.tambah_spasi()
            elif tombol in (8, 127):                # BACKSPACE
                penstabil.hapus_terakhir()
            elif tombol in (ord("r"), ord("R")):    # R = Reset
                penstabil.reset()
    finally:
        kamera.release()
        detektor.close()
        cv2.destroyAllWindows()
        print(f"Teks akhir: '{penstabil.teks}'")


if __name__ == "__main__":
    main()
