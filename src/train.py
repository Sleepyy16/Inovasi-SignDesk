"""Latih model pengenal huruf BISINDO dari semua CSV di data/landmarks/.

Langkah:
  1. Gabungkan semua CSV dan bersihkan baris rusak.
  2. Split latih/uji BERDASARKAN person_id (orang/dataset yang diuji tidak ada di data latih).
  3. Latih RandomForest (baseline) dan MLP, bandingkan akurasi uji.
  4. Simpan model terbaik + daftar huruf + info split ke models/.

Jalankan dari folder utama:  python src/train.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import features


def muat_data():
    """Gabungkan semua CSV landmark dan buang baris rusak. Kembalikan DataFrame."""
    file_csv = sorted(config.LANDMARKS_DIR.glob("*.csv"))
    if not file_csv:
        print(f"Folder data kosong: {config.LANDMARKS_DIR}")
        print("Jalankan dulu extract_from_images.py (T2) dan/atau collect_data.py (T3).")
        return None

    kolom_fitur = features.nama_kolom_fitur()
    kolom_standar = ["label", "person_id", "source"] + kolom_fitur
    bagian = []
    for file in file_csv:
        try:
            df = pd.read_csv(file)
            # Jika header memakai format lama (f_000..f_126), sesuaikan otomatis
            if "f_000" in df.columns and len(df.columns) == len(kolom_standar):
                df.columns = kolom_standar
            bagian.append(df)
            print(f"Dibaca: {file.name} ({len(df)} baris)")
        except Exception as error:
            print(f"PERINGATAN: {file.name} dilewati, gagal dibaca: {error}")
    if not bagian:
        return None

    tabel = pd.concat(bagian, ignore_index=True)
    kolom_wajib = ["label", "person_id"] + kolom_fitur
    hilang = [k for k in kolom_wajib if k not in tabel.columns]
    if hilang:
        print(f"CSV tidak punya kolom wajib (contoh: {hilang[:3]}). Ekstrak ulang datanya.")
        return None

    awal = len(tabel)
    tabel = tabel.replace([np.inf, -np.inf], np.nan)
    tabel = tabel.dropna(subset=kolom_wajib)
    tabel = tabel[tabel["label"].isin(config.HURUF)].reset_index(drop=True)
    tabel["person_id"] = tabel["person_id"].astype(str)
    if len(tabel) < awal:
        print(f"Dibuang {awal - len(tabel)} baris rusak/label tidak valid.")

    # Buang baris yang jumlah tangannya tidak sesuai mayoritas huruf tersebut
    # (contoh: foto huruf 2-tangan seperti H yang tangan keduanya gagal terdeteksi MediaPipe).
    if "jumlah_tangan" in tabel.columns:
        sebelum_filter = len(tabel)
        mayoritas_tangan = tabel.groupby("label")["jumlah_tangan"].transform(
            lambda s: s.round().mode().iloc[0]
        )
        tabel = tabel[tabel["jumlah_tangan"].round() == mayoritas_tangan].reset_index(drop=True)
        if len(tabel) < sebelum_filter:
            print(
                f"Dibuang {sebelum_filter - len(tabel)} baris yang jumlah tangannya "
                "tidak konsisten (gagal deteksi tangan ke-2)."
            )

    print(f"Total data bersih: {len(tabel)} baris, {tabel['label'].nunique()} huruf.")
    return tabel


def buat_split(tabel):
    """Kembalikan (indeks_latih, indeks_uji, mode, grup_uji).

    mode 'grup'  : split per person_id (yang benar).
    mode 'acak'  : cadangan jika person_id tidak informatif; hasil bisa TERLALU OPTIMIS.
    """
    grup = tabel["person_id"].values
    jumlah_grup = len(set(grup))
    semua_unknown = jumlah_grup == 1 and grup[0] == "unknown"

    if jumlah_grup < 2:
        print("\n" + "!" * 60)
        print("PERINGATAN KERAS: hanya ada 1 person_id" +
              (" ('unknown')" if semua_unknown else "") + ".")
        print("Split per penutur TIDAK mungkin. Memakai split ACAK sebagai cadangan.")
        print("Akurasi akan TERLALU OPTIMIS (foto orang yang sama ada di latih dan uji).")
        print("Jangan laporkan angka ini sebagai kemampuan di dunia nyata.")
        print("!" * 60 + "\n")
        indeks = np.arange(len(tabel))
        latih, uji = train_test_split(indeks, test_size=config.RASIO_UJI,
                                      random_state=config.RANDOM_SEED,
                                      stratify=tabel["label"].values)
        return latih, uji, "acak", []

    pemisah = GroupShuffleSplit(n_splits=1, test_size=config.RASIO_UJI,
                                random_state=config.RANDOM_SEED)
    latih, uji = next(pemisah.split(tabel, groups=grup))
    grup_uji = sorted(set(grup[uji]))
    return latih, uji, "grup", grup_uji


def terapkan_split_tersimpan(tabel, mode, grup_uji):
    """Ulangi split yang sama saat evaluasi (dipakai evaluate.py)."""
    if mode == "grup":
        uji = np.where(tabel["person_id"].isin(grup_uji))[0]
        latih = np.where(~tabel["person_id"].isin(grup_uji))[0]
        return latih, uji
    latih, uji, _, _ = buat_split(tabel)
    return latih, uji


def buat_model_rf():
    return RandomForestClassifier(n_estimators=config.RF_JUMLAH_POHON,
                                  random_state=config.RANDOM_SEED, n_jobs=-1)


def buat_model_mlp():
    # StandardScaler membantu MLP; Random Forest tidak membutuhkannya.
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=config.MLP_LAPISAN, max_iter=config.MLP_MAKS_ITERASI,
                      early_stopping=True, random_state=config.RANDOM_SEED),
    )


def main():
    tabel = muat_data()
    if tabel is None or tabel.empty:
        return

    kolom_fitur = features.nama_kolom_fitur()
    X = tabel[kolom_fitur].values.astype(np.float32)
    y = tabel["label"].values

    latih, uji, mode, grup_uji = buat_split(tabel)
    X_latih, y_latih = X[latih], y[latih]
    X_uji, y_uji = X[uji], y[uji]
    print(f"Split ({mode}): latih {len(latih)}, uji {len(uji)}")
    if mode == "grup":
        print(f"person_id di data UJI: {', '.join(grup_uji)}")
        print(f"person_id di data LATIH: {', '.join(sorted(set(tabel['person_id'].values[latih])))}")

    # Augmentasi cermin horizontal pada data latih:
    # Mengatasi perbedaan kamera mirror vs foto dataset sekaligus tangan kidal/kanan.
    X_latih = np.vstack([X_latih, features.cerminkan_fitur(X_latih)])
    y_latih = np.concatenate([y_latih, y_latih])
    print(f"Data latih setelah augmentasi cermin kiri-kanan: {len(X_latih)} sampel")

    tidak_ada_di_latih = sorted(set(y_uji) - set(y_latih))
    if tidak_ada_di_latih:
        print(f"PERINGATAN: huruf ada di uji tapi TIDAK di latih: {', '.join(tidak_ada_di_latih)} "
              "(pasti salah diprediksi).")
    tidak_ada_di_uji = sorted(set(config.HURUF) - set(y_uji))
    if tidak_ada_di_uji:
        print(f"Catatan: huruf tanpa data uji (tidak terukur): {', '.join(tidak_ada_di_uji)}")

    kandidat = {"RandomForest": buat_model_rf(), "MLP": buat_model_mlp()}
    hasil = {}
    for nama, model in kandidat.items():
        print(f"\nMelatih {nama} ...")
        model.fit(X_latih, y_latih)
        akurasi_latih = model.score(X_latih, y_latih)
        akurasi_uji = model.score(X_uji, y_uji)
        hasil[nama] = (akurasi_uji, akurasi_latih)
        print(f"  akurasi latih: {akurasi_latih:.3f} | akurasi uji: {akurasi_uji:.3f}")
        if akurasi_latih - akurasi_uji > 0.15:
            print("  ! Selisih besar: tanda overfitting / data uji berbeda dari data latih.")

    nama_terbaik = max(hasil, key=lambda n: hasil[n][0])
    print(f"\nModel terbaik: {nama_terbaik} (akurasi uji {hasil[nama_terbaik][0]:.3f})")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": kandidat[nama_terbaik],
        "nama_model": nama_terbaik,
        "labels": list(kandidat[nama_terbaik].classes_),
        "mode_split": mode,
        "grup_uji": grup_uji,
        "kolom_fitur": kolom_fitur,
    }, config.MODEL_PATH)
    print(f"Model disimpan: {config.MODEL_PATH}")
    print("Lanjut: python src/evaluate.py")


if __name__ == "__main__":
    main()
