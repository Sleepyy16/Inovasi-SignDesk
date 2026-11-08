"""Evaluasi model terlatih pada data uji (split yang sama dengan train.py).

Menampilkan: akurasi, classification report per huruf, akurasi per person_id,
5 pasangan huruf yang paling sering tertukar, dan menyimpan confusion matrix
sebagai gambar di reports/.

Jalankan dari folder utama:  python src/evaluate.py
(Jalankan train.py lebih dulu. Latih ulang jika data CSV berubah.)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")  # simpan gambar tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config
from train import muat_data, terapkan_split_tersimpan


def pasangan_tertukar(matriks, labels, jumlah=5):
    """Kembalikan daftar (benar, ditebak, jumlah) terbanyak, abaikan diagonal."""
    pasangan = []
    for i, benar in enumerate(labels):
        for j, tebak in enumerate(labels):
            if i != j and matriks[i, j] > 0:
                pasangan.append((benar, tebak, int(matriks[i, j])))
    pasangan.sort(key=lambda p: p[2], reverse=True)
    return pasangan[:jumlah]


def simpan_gambar_confusion(matriks, labels, akurasi):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    gambar, sumbu = plt.subplots(figsize=(11, 9))
    tampilan = sumbu.imshow(matriks, cmap="Blues")
    sumbu.set_xticks(range(len(labels)))
    sumbu.set_yticks(range(len(labels)))
    sumbu.set_xticklabels(labels)
    sumbu.set_yticklabels(labels)
    sumbu.set_xlabel("Prediksi")
    sumbu.set_ylabel("Huruf sebenarnya")
    sumbu.set_title(f"Confusion Matrix (akurasi uji {akurasi:.1%})")
    batas = matriks.max() / 2 if matriks.max() > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matriks[i, j] > 0:
                sumbu.text(j, i, int(matriks[i, j]), ha="center", va="center", fontsize=7,
                           color="white" if matriks[i, j] > batas else "black")
    gambar.colorbar(tampilan)
    gambar.tight_layout()
    try:
        with open(config.FILE_GAMBAR_CONFUSION, "wb") as file_out:
            gambar.savefig(file_out, format="png", dpi=150)
        print(f"Confusion matrix disimpan: {config.FILE_GAMBAR_CONFUSION}")
    except OSError as error:
        print(f"PERINGATAN: gagal menyimpan gambar confusion matrix ({error}).")
        print("Pastikan file confusion_matrix.png tidak sedang dibuka di aplikasi lain.")
    finally:
        plt.close(gambar)


def main():
    if not config.MODEL_PATH.exists():
        print(f"Model belum ada: {config.MODEL_PATH}")
        print("Jalankan dulu: python src/train.py")
        return
    paket = joblib.load(config.MODEL_PATH)
    model = paket["model"]

    tabel = muat_data()
    if tabel is None or tabel.empty:
        return
    kolom_fitur = paket["kolom_fitur"]
    _, uji = terapkan_split_tersimpan(tabel, paket["mode_split"], paket["grup_uji"])
    if len(uji) == 0:
        print("Data uji kosong. Latih ulang dengan train.py.")
        return

    X_uji = tabel[kolom_fitur].values.astype(np.float32)[uji]
    y_uji = tabel["label"].values[uji]
    id_uji = tabel["person_id"].values[uji]
    y_tebak = model.predict(X_uji)

    akurasi = accuracy_score(y_uji, y_tebak)
    print(f"\nModel: {paket['nama_model']} | mode split: {paket['mode_split']}")
    if paket["mode_split"] == "acak":
        print("PERINGATAN: split ACAK, akurasi terlalu optimis (lihat peringatan di train.py).")
    print(f"Data uji: {len(uji)} sampel | AKURASI: {akurasi:.3f}")

    labels = sorted(set(y_uji) | set(y_tebak))
    print("\n=== Laporan per huruf ===")
    print(classification_report(y_uji, y_tebak, labels=labels, zero_division=0))

    print("=== Akurasi per person_id (data uji) ===")
    for person in sorted(set(id_uji)):
        mask = id_uji == person
        print(f"  {person}: {accuracy_score(y_uji[mask], y_tebak[mask]):.3f} ({mask.sum()} sampel)")

    matriks = confusion_matrix(y_uji, y_tebak, labels=labels)
    print("\n=== 5 pasangan huruf paling sering tertukar ===")
    pasangan = pasangan_tertukar(matriks, labels)
    if not pasangan:
        print("  Tidak ada kesalahan (periksa apakah data uji terlalu mudah).")
    for benar, tebak, jumlah in pasangan:
        print(f"  {benar} sering ditebak sebagai {tebak}: {jumlah} kali")

    huruf_buruk = [(h, r) for h, r in zip(labels, matriks.diagonal() / np.maximum(matriks.sum(axis=1), 1)) if r < 0.7]
    if huruf_buruk:
        print("\nHuruf dengan recall < 70% (target README): " +
              ", ".join(f"{h} ({r:.0%})" for h, r in huruf_buruk))

    simpan_gambar_confusion(matriks, labels, akurasi)


if __name__ == "__main__":
    main()
