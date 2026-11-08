"""Konversi teks balasan petugas menjadi urutan gambar huruf isyarat BISINDO (fingerspelling).

Aturan sesuai spesifikasi:
- Ubah semua huruf menjadi HURUF BESAR (uppercase).
- Buang karakter yang tidak didukung (hanya simpan A-Z dan spasi sebagai pemisah kata).
- Ambil gambar tiap huruf dari folder assets/alfabet/ (A.png s/d Z.png).
- Jika assets/alfabet/<HURUF>.png belum ada, modul ini dapat menyalin otomatis
  1 gambar perwakilan per huruf dari data/raw_images/ atau menampilkan placeholder.

Jalankan langsung untuk menyiapkan gambar assets/alfabet/ dan menguji modul:
    python src/text_to_sign.py
"""

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2

import config


def bersihkan_teks(teks):
    """Ubah teks ke huruf besar dan saring hanya A-Z serta spasi tunggal.

    Kembalikan tuple:
      (teks_bersih, daftar_karakter_dibuang)
    """
    if not teks:
        return "", []

    teks_besar = str(teks).upper()
    karakter_valid = []
    dibuang = []

    for kar in teks_besar:
        if kar in config.HURUF:
            karakter_valid.append(kar)
        elif kar.isspace():
            karakter_valid.append(" ")
        else:
            if kar not in dibuang:
                dibuang.append(kar)

    # Rapikan spasi berlebih menjadi satu spasi antar kata
    teks_bersih = " ".join("".join(karakter_valid).split())
    return teks_bersih, dibuang


def cari_gambar_huruf(huruf):
    """Cari file gambar untuk 1 huruf di assets/alfabet/ (prioritas A.png).

    Jika belum ada di assets/alfabet/, cari cadangan di data/raw_images/.
    Kembalikan Path jika ketemu, atau None jika belum tersedia.
    """
    huruf = str(huruf).upper()
    if huruf not in config.HURUF:
        return None

    # 1. Cek di assets/alfabet/A.png (atau ekstensi gambar lain)
    prioritas_ekstensi = (".png",) + tuple(
        eks for eks in config.EKSTENSI_GAMBAR if eks != ".png"
    )
    for eks in prioritas_ekstensi:
        kandidat = config.ASSETS_ALFABET_DIR / f"{huruf}{eks}"
        if kandidat.exists():
            return kandidat

    # 2. Cadangan: cek di data/raw_images/<huruf>/ atau data/raw_images/*/<huruf>/
    if config.RAW_IMAGES_DIR.exists():
        folder_langsung = config.RAW_IMAGES_DIR / huruf
        calon_folder = [folder_langsung] if folder_langsung.is_dir() else []
        for sub in sorted(config.RAW_IMAGES_DIR.iterdir()):
            if sub.is_dir() and (sub / huruf).is_dir():
                calon_folder.append(sub / huruf)

        for folder in calon_folder:
            for file in sorted(folder.iterdir()):
                if file.is_file() and file.suffix.lower() in config.EKSTENSI_GAMBAR:
                    return file

    return None


def siapkan_aset_alfabet():
    """Salin/konversi 1 gambar per huruf (A-Z) dari data/raw_images/ ke assets/alfabet/A.png..Z.png.

    Hanya mengisi huruf yang belum punya file di assets/alfabet/.
    Kembalikan jumlah huruf yang siap di assets/alfabet/.
    """
    config.ASSETS_ALFABET_DIR.mkdir(parents=True, exist_ok=True)
    jumlah_siap = 0

    for huruf in config.HURUF:
        target_png = config.ASSETS_ALFABET_DIR / f"{huruf}.png"
        if target_png.exists():
            jumlah_siap += 1
            continue

        sumber = cari_gambar_huruf(huruf)
        if sumber is None:
            continue

        try:
            if sumber.suffix.lower() == ".png":
                shutil.copyfile(sumber, target_png)
                jumlah_siap += 1
            else:
                gambar = cv2.imread(str(sumber))
                if gambar is not None:
                    berhasil, buffer = cv2.imencode(".png", gambar)
                    if berhasil:
                        with open(target_png, "wb") as f:
                            f.write(buffer.tobytes())
                        jumlah_siap += 1
        except OSError:
            # Jika gagal menulis ke disk, fungsi cari_gambar_huruf tetap bisa memakai sumber asli
            pass

    return jumlah_siap


def teks_ke_isyarat(teks):
    """Ubah string teks menjadi daftar item isyarat untuk ditampilkan di UI.

    Kembalikan dict:
      - teks_bersih      : string hasil pembersihan (huruf besar)
      - karakter_dibuang : list karakter yang diabaikan (angka/tanda baca)
      - kata_isyarat     : list per kata, tiap kata berisi list dict:
                           {"huruf": 'A', "path_gambar": Path/None, "eksperimental": bool}
    """
    teks_bersih, dibuang = bersihkan_teks(teks)
    if not teks_bersih:
        return {
            "teks_bersih": "",
            "karakter_dibuang": dibuang,
            "kata_isyarat": [],
        }

    daftar_kata = []
    for kata in teks_bersih.split(" "):
        item_kata = []
        for huruf in kata:
            item_kata.append({
                "huruf": huruf,
                "path_gambar": cari_gambar_huruf(huruf),
                "eksperimental": huruf in config.HURUF_EKSPERIMENTAL,
            })
        daftar_kata.append(item_kata)

    return {
        "teks_bersih": teks_bersih,
        "karakter_dibuang": dibuang,
        "kata_isyarat": daftar_kata,
    }


def uji_text_to_sign():
    """Siapkan aset A.png-Z.png jika belum ada, lalu uji fungsi teks_ke_isyarat."""
    print("=== Menyiapkan & Menguji Modul Teks -> Isyarat ===")
    siap = siapkan_aset_alfabet()
    print(f"Gambar huruf tersedia di {config.ASSETS_ALFABET_DIR}: {siap}/26 huruf")

    contoh = "Halo 123, Silakan!"
    hasil = teks_ke_isyarat(contoh)
    print(f"Input asli       : {contoh!r}")
    print(f"Teks bersih      : {hasil['teks_bersih']!r}")
    print(f"Karakter dibuang : {hasil['karakter_dibuang']}")
    for idx, kata in enumerate(hasil["kata_isyarat"], start=1):
        ringkas = [
            f"{item['huruf']}({'ADA' if item['path_gambar'] else 'PLACEHOLDER'})"
            for item in kata
        ]
        print(f"  Kata {idx}: {' -> '.join(ringkas)}")

    assert hasil["teks_bersih"] == "HALO SILAKAN"
    assert "1" in hasil["karakter_dibuang"] and "!" in hasil["karakter_dibuang"]
    print("\n[OK] Pengujian text_to_sign BERHASIL!")


if __name__ == "__main__":
    uji_text_to_sign()
