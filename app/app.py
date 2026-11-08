"""Aplikasi Streamlit SignDesk — Mode Loket Dua Sisi (Pengunjung & Petugas).

Fitur Utama:
1. Sisi Pengunjung (Isyarat -> Teks):
   - Kamera real-time (streamlit-webrtc) dengan tampilan mirror & titik landmark.
   - Prediksi huruf BISINDO + skor keyakinan + bar stabilisasi 1 detik.
   - Kotak teks hasil ejaan beserta tombol Spasi, Hapus, Reset, dan Salin.
2. Sisi Petugas (Teks -> Isyarat):
   - Input teks balasan petugas.
   - Tombol "Tampilkan sebagai isyarat" yang mengubah teks menjadi urutan gambar
     huruf BISINDO (fingerspelling) dari folder assets/alfabet/.

PRIVASI & ETIKA:
- Video diproses langsung di memori perangkat dan TIDAK disimpan ke disk.
- Aplikasi tidak menyimpan nama atau data pribadi pengunjung.
- Alat bantu komunikasi dasar, bukan pengganti juru bahasa isyarat.

Jalankan dari folder utama:
    streamlit run app/app.py
"""

import sys
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in (str(BASE_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import av
import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import config
import features
from stabilizer import PenstabilPrediksi
import text_to_sign

DISCLAIMER_UTAMA = "Alat bantu komunikasi dasar, bukan pengganti juru bahasa isyarat."
INFO_PRIVASI = (
    "Privasi terjaga: Video kamera diproses secara lokal di perangkat ini dan TIDAK disimpan ke disk. "
    "Aplikasi tidak menyimpan nama atau data pribadi pengunjung."
)


class StatusLoketBersama:
    """Penyimpanan thread-safe antara thread video WebRTC dan antarmuka Streamlit."""

    def __init__(self):
        self.kunci = threading.Lock()
        self.penstabil = PenstabilPrediksi()
        self.huruf_tampil = "-"
        self.keyakinan = 0.0
        self.progres = 0.0
        self.ada_tangan = False
        self.detektor = None

    def ambil_detektor(self):
        if self.detektor is None:
            self.detektor = features.buat_detektor(gambar_statis=False)
        return self.detektor

    def perbarui_dari_frame(self, huruf_mentah, keyakinan, ada_tangan):
        with self.kunci:
            status = self.penstabil.perbarui(huruf_mentah, keyakinan)
            self.huruf_tampil = status["huruf_stabil"] or (huruf_mentah if huruf_mentah else "-")
            self.keyakinan = float(keyakinan)
            self.progres = float(status["progres"])
            self.ada_tangan = bool(ada_tangan)
            return dict(
                huruf_tampil=self.huruf_tampil,
                keyakinan=self.keyakinan,
                progres=self.progres,
                teks=status["teks"],
                ada_tangan=self.ada_tangan,
            )

    def baca_status(self):
        with self.kunci:
            return dict(
                huruf_tampil=self.huruf_tampil,
                keyakinan=self.keyakinan,
                progres=self.progres,
                teks=self.penstabil.teks,
                ada_tangan=self.ada_tangan,
            )

    def tambah_spasi(self):
        with self.kunci:
            self.penstabil.tambah_spasi()

    def hapus_terakhir(self):
        with self.kunci:
            self.penstabil.hapus_terakhir()

    def reset(self):
        with self.kunci:
            self.penstabil.reset()
            self.huruf_tampil = "-"
            self.keyakinan = 0.0
            self.progres = 0.0


@st.cache_resource
def muat_status_bersama():
    """Buat objek StatusLoketBersama satu kali agar bertahan lintas rerun Streamlit."""
    text_to_sign.siapkan_aset_alfabet()
    return StatusLoketBersama()


@st.cache_resource
def muat_model_terlatih():
    """Muat model dari models/model_terbaik.joblib. Kembalikan (model, nama_model, pesan_error)."""
    if not config.MODEL_PATH.exists():
        return (
            None,
            None,
            f"File model belum ditemukan di `{config.MODEL_PATH}`. "
            "Jalankan `python src/train.py` terlebih dahulu di terminal.",
        )
    try:
        paket = joblib.load(config.MODEL_PATH)
        return paket["model"], paket.get("nama_model", "Model"), None
    except Exception as error:
        return None, None, f"Gagal memuat model: {error}. Coba jalankan ulang `python src/train.py`."


def prediksi_huruf(model, vektor):
    """Hitung (huruf, keyakinan) dari vektor fitur 127 angka."""
    if model is None or vektor is None:
        return None, 0.0
    X = np.asarray(vektor, dtype=np.float32).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        peluang = model.predict_proba(X)[0]
        idx = int(np.argmax(peluang))
        return str(model.classes_[idx]), float(peluang[idx])
    return str(model.predict(X)[0]), 1.0


def gambar_overlay_video(img, landmark, info):
    """Gambar titik landmark, prediksi huruf, bar stabilisasi, dan teks di frame video."""
    tinggi, lebar = img.shape[:2]
    for tangan in landmark:
        for x, y, _ in tangan:
            cv2.circle(img, (int(x * lebar), int(y * tinggi)), 4, (0, 255, 0), -1)

    if config.CERMINKAN_KAMERA and not config.FITUR_DARI_FRAME_CERMIN:
        img = cv2.flip(img, 1)

    # Latar bar atas & bawah agar tulisan mudah dibaca
    lapisan = img.copy()
    cv2.rectangle(lapisan, (0, 0), (lebar, 82), (20, 20, 20), -1)
    cv2.rectangle(lapisan, (0, tinggi - 60), (lebar, tinggi), (20, 20, 20), -1)
    cv2.addWeighted(lapisan, 0.72, img, 0.28, 0, img)

    huruf = info["huruf_tampil"]
    persen = int(round(info["keyakinan"] * 100))
    warna = (0, 255, 100) if info["keyakinan"] >= config.AMBANG_KEYAKINAN else (0, 200, 255)
    label_atas = f"Huruf: {huruf} ({persen}%)"
    if huruf in config.HURUF_EKSPERIMENTAL:
        label_atas += " [eksperimental]"
    cv2.putText(img, label_atas, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna, 2)

    # Bar progres stabilisasi 1 detik
    lebar_maks = 220
    isi = int(lebar_maks * info["progres"])
    cv2.rectangle(img, (14, 46), (14 + lebar_maks, 66), (120, 120, 120), 1)
    if isi > 0:
        cv2.rectangle(img, (14, 46), (14 + isi, 66), (0, 255, 120), -1)

    potongan = info["teks"][-28:] if len(info["teks"]) > 28 else info["teks"]
    cv2.putText(img, f"Teks: {potongan}_", (14, tinggi - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return img


def tampilkan_panel_teks_pengunjung(status_bersama):
    """Panel metrik dan kontrol teks pengunjung (mendukung auto-refresh jika tersedia)."""
    info = status_bersama.baca_status()

    kol_m1, kol_m2, kol_m3 = st.columns(3)
    label_huruf = info["huruf_tampil"]
    if label_huruf in config.HURUF_EKSPERIMENTAL:
        label_huruf += " *"
    kol_m1.metric("Huruf Terbaca", label_huruf)
    kol_m2.metric("Keyakinan", f"{int(round(info['keyakinan'] * 100))}%")
    kol_m3.metric("Status Tangan", "Terdeteksi" if info["ada_tangan"] else "Tidak ada")

    st.progress(min(1.0, max(0.0, info["progres"])), text="Progres tahan ~1 detik untuk mengetik huruf")

    st.markdown("**Teks Hasil Isyarat Pengunjung:**")
    teks_sekarang = info["teks"]
    st.info(teks_sekarang if teks_sekarang else "(Belum ada huruf yang masuk — tahan isyarat ~1 detik di depan kamera)")

    t1, t2, t3, t4 = st.columns(4)
    if t1.button("Spasi", use_container_width=True):
        status_bersama.tambah_spasi()
        st.rerun()
    if t2.button("Hapus", use_container_width=True):
        status_bersama.hapus_terakhir()
        st.rerun()
    if t3.button("Reset", use_container_width=True):
        status_bersama.reset()
        st.rerun()
    if t4.button("Salin", use_container_width=True):
        st.session_state["tampilkan_salin"] = True

    if st.session_state.get("tampilkan_salin"):
        st.caption("Klik ikon salin di pojok kanan kotak di bawah ini:")
        st.code(teks_sekarang or "", language=None)


def tampilkan_kolom_petugas():
    """Kolom kanan: input balasan petugas dan tampilan urutan gambar isyarat (fingerspelling)."""
    st.subheader("Sisi Petugas (Teks -> Isyarat)")
    st.caption("Ketik balasan singkat untuk pengunjung, lalu tampilkan sebagai urutan huruf isyarat BISINDO.")

    with st.form("form_balasan_petugas", clear_on_submit=False):
        input_teks = st.text_input(
            "Ketik balasan petugas (huruf A-Z):",
            value=st.session_state.get("teks_petugas_terakhir", "HALO SILAKAN"),
            max_chars=60,
            placeholder="Contoh: NAMA SIAPA",
        )
        tombol_tampil = st.form_submit_button("Tampilkan sebagai isyarat", use_container_width=True)

    if tombol_tampil:
        st.session_state["teks_petugas_terakhir"] = input_teks

    teks_aktif = st.session_state.get("teks_petugas_terakhir", input_teks)
    hasil = text_to_sign.teks_ke_isyarat(teks_aktif)

    if hasil["karakter_dibuang"]:
        dibuang_str = ", ".join(repr(k) for k in hasil["karakter_dibuang"])
        st.warning(f"Karakter selain A-Z diabaikan otomatis: {dibuang_str}")

    if not hasil["kata_isyarat"]:
        st.info("Masukkan minimal 1 huruf (A-Z) untuk menampilkan gambar isyarat.")
        return

    st.markdown(f"**Ejaan Isyarat:** `{hasil['teks_bersih']}`")

    ada_eksperimental = False
    Kolom_Per_Baris = 6
    for indeks_kata, kata in enumerate(hasil["kata_isyarat"], start=1):
        huruf_kata = "".join(item["huruf"] for item in kata)
        st.markdown(f"**Kata {indeks_kata}: {huruf_kata}**")

        for mulai in range(0, len(kata), Kolom_Per_Baris):
            potongan = kata[mulai : mulai + Kolom_Per_Baris]
            kolom_gambar = st.columns(Kolom_Per_Baris)
            for kol, item in zip(kolom_gambar, potongan):
                with kol:
                    label_huruf = item["huruf"]
                    if item["eksperimental"]:
                        ada_eksperimental = True
                        label_huruf += " (eks.)"
                    if item["path_gambar"] is not None and Path(item["path_gambar"]).exists():
                        st.image(str(item["path_gambar"]), caption=label_huruf, use_container_width=True)
                    else:
                        st.markdown(
                            f"<div style='border:1px dashed #888; border-radius:8px; padding:18px 6px; "
                            f"text-align:center; background:#222; color:#eee;'>"
                            f"<b style='font-size:1.4em;'>{item['huruf']}</b><br>"
                            f"<small>letakkan {item['huruf']}.png di assets/alfabet/</small></div>",
                            unsafe_allow_html=True,
                        )

    if ada_eksperimental:
        st.caption(
            "*Catatan: Huruf **J** dan **Z** pada BISINDO aslinya melibatkan gerakan; "
            "pada MVP ini ditampilkan sebagai pose statis (eksperimental)."
        )


def main():
    st.set_page_config(
        page_title="SignDesk — Mode Loket BISINDO",
        layout="wide",
    )

    st.title("SignDesk — Mode Loket Komunikasi Alfabet BISINDO")
    st.warning(f"**Disclaimer:** {DISCLAIMER_UTAMA}")
    st.caption(INFO_PRIVASI)

    status_bersama = muat_status_bersama()
    model, nama_model, pesan_error_model = muat_model_terlatih()

    kol_pengunjung, kol_petugas = st.columns(2, gap="large")

    with kol_pengunjung:
        st.subheader("Sisi Pengunjung (Isyarat -> Teks)")
        if pesan_error_model:
            st.error(pesan_error_model)
            st.info(
                "Kamera belum diaktifkan karena model belum tersedia. "
                "Setelah menjalankan `python src/train.py`, muat ulang (refresh) halaman ini."
            )
        else:
            st.caption(
                f"Model aktif: **{nama_model}** | Ambang keyakinan: **{int(config.AMBANG_KEYAKINAN * 100)}%** | "
                "Huruf **J** & **Z** bertanda *(eksperimental)*."
            )

            def callback_frame(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                img = features.siapkan_frame_untuk_deteksi(img)
                detektor = status_bersama.ambil_detektor()
                vektor, landmark = features.ekstrak_fitur_dan_landmark(img, detektor)
                huruf_mentah, keyakinan = prediksi_huruf(model, vektor)
                info = status_bersama.perbarui_dari_frame(
                    huruf_mentah, keyakinan, ada_tangan=(vektor is not None)
                )
                img = gambar_overlay_video(img, landmark, info)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(
                key="signdesk-webcam",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=callback_frame,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # Gunakan st.fragment(run_every=0.5) jika versi Streamlit mendukung agar teks otomatis terbarui
            if hasattr(st, "fragment"):
                panel_otomatis = st.fragment(run_every=0.5)(tampilkan_panel_teks_pengunjung)
                panel_otomatis(status_bersama)
            else:
                if st.button("Segarkan Teks dari Kamera"):
                    st.rerun()
                tampilkan_panel_teks_pengunjung(status_bersama)

    with kol_petugas:
        tampilkan_kolom_petugas()

    st.divider()
    st.caption(
        f"{DISCLAIMER_UTAMA} | Fase MVP: Alfabet BISINDO (A-Z), mendukung isyarat 1 & 2 tangan."
    )


if __name__ == "__main__":
    main()
