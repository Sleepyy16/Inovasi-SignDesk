"""
SignDesk - src/train.py
======================
Skrip pelatihan model machine learning untuk klasifikasi alfabet BISINDO.

ALUR KERJA:
1. Menggabungkan semua file CSV di folder data/landmarks/
   (baik dari dataset gambar maupun rekaman webcam).
2. Membersihkan data (memvalidasi 127 fitur, membuang baris rusak/NaN).
3. Pemisahan Data (Split):
   - Jika tersedia person_id unik (bukan 'unknown'): menggunakan GroupShuffleSplit
     agar penutur pada data uji TIDAK PERNAH muncul pada data latih (speaker-independent).
   - Jika semua data 'unknown': menggunakan stratified train_test_split dengan
     peringatan keras potensi over-optimistic / leak.
4. Melatih 2 Model:
   - Model A: RandomForestClassifier (cepat, stabil, tahan noise).
   - Model B: MLPClassifier (jaringan saraf tiruan multi-layer perceptron).
5. Membandingkan performa latih vs uji (mendeteksi tanda-tanda overfitting).
6. Menyimpan model terbaik & metadata label ke folder models/.

CARA MENJALANKAN:
  python src/train.py
"""

import sys
import os
import glob
import tempfile
import time

# Hindari peringatan direktori cache matplotlib yang tidak writable di Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Tambahkan root proyek ke sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import config
from src.features import nama_kolom_fitur


def muat_dan_gabungkan_data() -> pd.DataFrame:
    """
    Mencari dan menggabungkan seluruh file CSV di folder data/landmarks/.
    """
    pola_csv = os.path.join(config.LANDMARKS_DIR, "*.csv")
    daftar_file = glob.glob(pola_csv)

    # Kecualikan file gabungan lama jika ada agar tidak terduplikasi
    daftar_file = [
        f for f in daftar_file
        if os.path.basename(f) != os.path.basename(config.LANDMARKS_CSV)
    ]

    if not daftar_file:
        print(f"❌ Tidak ditemukan file CSV data landmark di '{config.LANDMARKS_DIR}'.")
        print("   Jalankan salah satu skrip ekstraksi data terlebih dahulu:")
        print("   - python src/extract_from_images.py  (dari dataset citra)")
        print("   - python src/collect_data.py         (dari webcam)")
        sys.exit(1)

    print(f"📂 Ditemukan {len(daftar_file)} file CSV data:")
    list_df = []
    for f in daftar_file:
        try:
            df_part = pd.read_csv(f)
            print(f"   - {os.path.basename(f)}: {len(df_part)} baris")
            list_df.append(df_part)
        except Exception as e:
            print(f"   ⚠️ Gagal membaca {os.path.basename(f)}: {e}")

    if not list_df:
        print("❌ Seluruh file CSV kosong atau tidak bisa dibaca.")
        sys.exit(1)

    df_gabungan = pd.concat(list_df, ignore_index=True)
    return df_gabungan


def bersihkan_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Membersihkan data:
    - Menghapus baris yang memiliki nilai NaN/kosong
    - Memastikan label huruf kapital
    - Memastikan jumlah kolom fitur persis 127 angka (f_000 s.d. f_126)
    """
    baris_awal = len(df)
    kolom_fitur = nama_kolom_fitur()

    # Periksa ketersediaan kolom wajib
    kolom_wajib = ["label", "person_id"] + kolom_fitur
    kolom_hilang = [k for k in kolom_wajib if k not in df.columns]
    if kolom_hilang:
        print(f"❌ Format CSV tidak lengkap! Kolom berikut hilang: {kolom_hilang[:5]}...")
        sys.exit(1)

    # Buang baris dengan nilai kosong (NaN / None)
    df_bersih = df.dropna(subset=kolom_wajib).copy()

    # Pastikan label huruf kapital dan string rapi
    df_bersih["label"] = df_bersih["label"].astype(str).str.strip().str.upper()
    df_bersih["person_id"] = df_bersih["person_id"].astype(str).str.strip()

    # Buang label yang bukan huruf tunggal alfabet A-Z
    df_bersih = df_bersih[df_bersih["label"].isin(config.HURUF_DIDUKUNG)].copy()

    baris_terbuang = baris_awal - len(df_bersih)
    if baris_terbuang > 0:
        print(f"🧹 Membersihkan data: {baris_terbuang} baris rusak/tidak valid dibuang.")

    print(f"✅ Total data bersih siap latih: {len(df_bersih)} sampel")

    # Pisahkan matriks fitur (X), target label (y), dan ID penutur (groups)
    X = df_bersih[kolom_fitur].values.astype(np.float32)
    y = df_bersih["label"].values
    groups = df_bersih["person_id"].values

    return X, y, groups, df_bersih


def lakukan_split_data(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """
    Melakukan pembagian data latih (train) dan uji (test).
    Prioritas 1: Split berdasarkan person_id (GroupShuffleSplit).
    Prioritas 2: Jika semua 'unknown', gunakan Stratified Train-Test Split.
    """
    penutur_unik = set(groups)
    ada_penutur_diketahui = any(p.lower() != "unknown" for p in penutur_unik)

    # Skenario 1: Ada minimal 2 penutur yang diketahui
    if ada_penutur_diketahui and len(penutur_unik) >= 2:
        print("\n🛡️ METODE SPLIT: Speaker-Independent (GroupShuffleSplit)")
        print(f"   Ditemukan {len(penutur_unik)} penutur unik: {sorted(penutur_unik)}")
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=config.PROPORSI_UJI,
            random_state=config.RANDOM_SEED
        )
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        penutur_train = set(groups[train_idx])
        penutur_test = set(groups[test_idx])
        print(f"   - Penutur di Data Latih : {sorted(penutur_train)}")
        print(f"   - Penutur di Data Uji   : {sorted(penutur_test)}")
        metode_split = "speaker_independent"

    # Skenario 2: Semua penutur 'unknown' atau hanya 1 penutur
    else:
        print("\n⚠️ PERINGATAN KERAS TENTANG VALIDITAS EVALUASI:")
        print("   Semua data bernilai person_id = 'unknown' (atau hanya ada 1 penutur).")
        print("   Evaluasi per-penutur (speaker-independent) belum dapat dilakukan!")
        print("   Menggunakan fallback: Stratified Random Split.")
        print("   ⚠️ Catatan: Model mungkin terlihat sangat akurat di pengujian ini,")
        print("   tetapi bisa turun akurasinya saat dicoba orang baru di dunia nyata.")
        print("   💡 Disarankan merekam data penutur lain di 'src/collect_data.py'.\n")

        # Cek apakah setiap kelas punya minimal 2 sampel untuk stratify
        kelas_counts = pd.Series(y).value_counts()
        kelas_langka = kelas_counts[kelas_counts < 2].index.tolist()

        if len(kelas_langka) > 0:
            print(f"   ℹ️ Huruf {kelas_langka} memiliki kurang dari 2 sampel. Split acak non-stratified digunakan.")
            idx_semua = np.arange(len(X))
            train_idx, test_idx = train_test_split(
                idx_semua,
                test_size=config.PROPORSI_UJI,
                random_state=config.RANDOM_SEED,
                shuffle=True
            )
        else:
            idx_semua = np.arange(len(X))
            train_idx, test_idx = train_test_split(
                idx_semua,
                test_size=config.PROPORSI_UJI,
                random_state=config.RANDOM_SEED,
                stratify=y,
                shuffle=True
            )
        metode_split = "random_stratified"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups[test_idx]

    print(f"📊 Pembagian Data: {len(X_train)} Latih ({(len(X_train)/len(X)*100):.1f}%) | "
          f"{len(X_test)} Uji ({(len(X_test)/len(X)*100):.1f}%)")

    return X_train, X_test, y_train, y_test, groups_test, metode_split


def latih_dan_bandingkan_model(X_train, X_test, y_train, y_test):
    """
    Melatih RandomForest dan MLPClassifier, lalu membandingkan performanya.
    """
    print("\n" + "=" * 65)
    print("🤖 PROSES PELATIHAN MODEL MACHINE LEARNING")
    print("=" * 65)

    hasil_model = {}

    # --- 1. Random Forest Classifier ---
    print("1️⃣ Melatih RandomForestClassifier...")
    waktu_awal = time.time()
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    durasi_rf = time.time() - waktu_awal

    acc_train_rf = accuracy_score(y_train, rf.predict(X_train))
    acc_test_rf = accuracy_score(y_test, rf.predict(X_test))
    f1_test_rf = f1_score(y_test, rf.predict(X_test), average="macro")

    hasil_model["RandomForest"] = {
        "model": rf,
        "acc_train": acc_train_rf,
        "acc_test": acc_test_rf,
        "f1_macro": f1_test_rf,
        "durasi": durasi_rf
    }
    print(f"   ✓ Selesai ({durasi_rf:.2f} detik) | Akurasi Latih: {acc_train_rf*100:.1f}% | Akurasi Uji: {acc_test_rf*100:.1f}%")

    # --- 2. Multi-Layer Perceptron (MLP) ---
    print("\n2️⃣ Melatih MLPClassifier (Jaringan Saraf Tiruan)...")
    waktu_awal = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=config.MLP_HIDDEN_LAYER_SIZES,
        max_iter=config.MLP_MAX_ITER,
        random_state=config.RANDOM_SEED,
        early_stopping=True,
        n_iter_no_change=15
    )
    mlp.fit(X_train, y_train)
    durasi_mlp = time.time() - waktu_awal

    acc_train_mlp = accuracy_score(y_train, mlp.predict(X_train))
    acc_test_mlp = accuracy_score(y_test, mlp.predict(X_test))
    f1_test_mlp = f1_score(y_test, mlp.predict(X_test), average="macro")

    hasil_model["MLP"] = {
        "model": mlp,
        "acc_train": acc_train_mlp,
        "acc_test": acc_test_mlp,
        "f1_macro": f1_test_mlp,
        "durasi": durasi_mlp
    }
    print(f"   ✓ Selesai ({durasi_mlp:.2f} detik) | Akurasi Latih: {acc_train_mlp*100:.1f}% | Akurasi Uji: {acc_test_mlp*100:.1f}%")

    # --- Tabel Perbandingan & Analisis Overfitting ---
    print("\n" + "=" * 65)
    print("📈 HASIL PERBANDINGAN MODEL")
    print("=" * 65)
    print(f"{'Nama Model':<16} | {'Akurasi Latih':<14} | {'Akurasi Uji':<12} | {'F1 Macro':<10} | {'Waktu':<8}")
    print("-" * 65)
    for nama, res in hasil_model.items():
        print(f"{nama:<16} | {res['acc_train']*100:>12.2f}% | {res['acc_test']*100:>10.2f}% | {res['f1_macro']:>8.4f} | {res['durasi']:>5.2f}s")
    print("-" * 65)

    # Deteksi Overfitting Sederhana
    for nama, res in hasil_model.items():
        gap = res['acc_train'] - res['acc_test']
        if gap > 0.15:
            print(f"⚠️ Catatan {nama}: Selisih latih vs uji sebesar {gap*100:.1f}%. "
                  f"Terindikasi overfitting (model menghafal data latih).")

    # Tentukan model terbaik berdasarkan akurasi uji (atau F1-score jika seri)
    if hasil_model["RandomForest"]["acc_test"] >= hasil_model["MLP"]["acc_test"]:
        pemenang = "RandomForest"
    else:
        pemenang = "MLP"

    print(f"\n🏆 Model Terbaik: {pemenang} (Akurasi Uji: {hasil_model[pemenang]['acc_test']*100:.2f}%)")

    return hasil_model[pemenang]["model"], pemenang, hasil_model


def simpan_model_dan_data_uji(model, nama_model: str, X_test, y_test, groups_test, label_unik, metode_split):
    """
    Menyimpan model terbaik, daftar kelas, dan split data uji untuk dievaluasi oleh evaluate.py.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # 1. Simpan model terlatih
    joblib.dump(model, config.MODEL_PATH)
    print(f"\n💾 Model tersimpan ke : {config.MODEL_PATH}")

    # 2. Simpan daftar label alfabet
    joblib.dump(label_unik, config.MODEL_LABEL_PATH)
    print(f"🏷️  Daftar label ke   : {config.MODEL_LABEL_PATH}")

    # 3. Simpan data uji (test set) untuk dipakai modul evaluasi
    test_set_path = os.path.join(config.MODELS_DIR, "data_uji_terakhir.joblib")
    paket_uji = {
        "X_test": X_test,
        "y_test": y_test,
        "groups_test": groups_test,
        "nama_model": nama_model,
        "metode_split": metode_split
    }
    joblib.dump(paket_uji, test_set_path)
    print(f"📦 Data uji tersimpan : {test_set_path}")


def main():
    print("=" * 65)
    print("  SignDesk — Pipeline Pelatihan Model Alfabet BISINDO")
    print("=" * 65)

    # 1. Muat & gabungkan data
    df_raw = muat_dan_gabungkan_data()

    # 2. Bersihkan data
    X, y, groups, df_bersih = bersihkan_data(df_raw)

    # 3. Lakukan split latih/uji
    X_train, X_test, y_train, y_test, groups_test, metode_split = lakukan_split_data(X, y, groups)

    # 4. Latih dan bandingkan model
    model_terbaik, nama_pemenang, _ = latih_dan_bandingkan_model(X_train, X_test, y_train, y_test)

    # 5. Simpan model dan data uji
    label_unik = sorted(list(set(y)))
    simpan_model_dan_data_uji(
        model=model_terbaik,
        nama_model=nama_pemenang,
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        label_unik=label_unik,
        metode_split=metode_split
    )

    print("\n🎉 Pelatihan selesai! Lanjutkan ke tahap evaluasi mendalam:")
    print("   👉 Jalankan: python src/evaluate.py\n")


if __name__ == "__main__":
    main()

