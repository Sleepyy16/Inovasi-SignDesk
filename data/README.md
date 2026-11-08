# ============================================================
# SignDesk - data/README.md
# Dokumentasi sumber data, lisensi, dan cara mengunduh.
# ============================================================

# Data SignDesk

## Isi Folder ini

```
data/
├── raw_images/      ← gambar dataset (TIDAK di-commit ke Git)
├── landmarks/       ← CSV hasil ekstraksi landmark (di-commit)
└── README.md        ← dokumen ini
```

## Sumber Data yang Digunakan

> **WAJIB diisi oleh anggota tim sebelum commit!**
> Catat setiap dataset yang dipakai, termasuk lisensi dan cara sitasi.

| Sumber | Nama Dataset | URL / DOI | Lisensi | Sitasi |
|--------|-------------|-----------|---------|--------|
| Citra BISINDO (Dataset Lokal) | Citra BISINDO (312 gambar A-Z) | Folder `DataSet/Citra BISINDO` | Riset Akademik | Tim / Sumber Terkait |
| Rekaman sendiri | Data internal webcam (`landmarks_kamera.csv`) | - | Milik tim, sudah ada persetujuan | - |

## Cara Mengunduh Dataset Publik

### 1. Dataset Kaggle (BISINDO Alphabets)

1. Buat akun Kaggle di https://kaggle.com
2. Pasang Kaggle CLI: `pip install kaggle`
3. Unduh API key dari profil Kaggle → letakkan di `~/.kaggle/kaggle.json`
4. Unduh dataset:
   ```bash
   kaggle datasets download -d achmadnoer/alfabet-bisindo
   # atau cari dengan: kaggle datasets list --search "bisindo"
   ```
5. Ekstrak ke `data/raw_images/`

### 2. Dataset Mendeley

1. Buka halaman dataset di https://data.mendeley.com
2. Unduh manual (tidak butuh akun)
3. Ekstrak ke `data/raw_images/`

## Struktur `raw_images` yang Diharapkan

Script `src/extract_from_images.py` mengasumsikan struktur berikut:

```
data/raw_images/
├── A/
│   ├── gambar_001.jpg
│   ├── gambar_002.jpg
│   └── ...
├── B/
│   └── ...
...
└── Z/
    └── ...
```

Jika dataset kamu punya struktur berbeda, baca komentar di awal
`src/extract_from_images.py` — ada instruksi cara menyesuaikannya.

## Format CSV Landmark

File di `data/landmarks/` punya kolom berikut:

| Kolom | Tipe | Keterangan |
|-------|------|-----------|
| `label` | string | Huruf A–Z |
| `person_id` | string | ID penutur, misal P01, P02, atau "unknown" |
| `source` | string | Sumber data, misal "kaggle", "mendeley", "rekaman_sendiri" |
| `f_000` s.d. `f_126` | float | 127 nilai fitur hasil ekstraksi |

## Persetujuan Pengumpulan Data

Setiap orang yang direkam **harus** menandatangani atau menyetujui secara
lisan bahwa:

- Data yang disimpan hanya angka koordinat tangan (bukan video/wajah)
- Data dipakai hanya untuk penelitian dan lomba ini
- Sewaktu-waktu dapat meminta datanya dihapus

Template persetujuan: *(buat dan lampirkan di sini)*

## Privasi

- ✅ Hanya landmark numerik yang disimpan, **bukan** gambar atau wajah
- ✅ Tidak ada identitas pribadi di CSV
- ✅ Folder `raw_images/` tidak masuk ke Git (ada di `.gitignore`)
