"""Penstabil prediksi huruf real-time agar teks tidak berkedip/loncat-loncat.

Masalah di kamera real-time:
- Dalam 1 detik ada ~30 frame. Kadang 1-2 frame salah tebak karena tangan bergerak.
- Jika setiap frame langsung dimasukkan ke teks, hasilnya akan berantakan (misal "AAAAABAAA").

Solusi di modul ini (PenstabilPrediksi):
1. Abaikan frame yang skor keyakinannya di bawah AMBANG_KEYAKINAN (default 0.7).
2. Simpan N prediksi terakhir (default 10 frame) dan ambil huruf terbanyak (voting).
3. Huruf baru masuk ke teks HANYA jika bertahan stabil selama DETIK_STABIL (default 1.0 detik).
4. Setelah huruf masuk, berikan jeda DETIK_JEDA_HURUF_SAMA (default 1.0 detik) sebelum
   huruf yang SAMA boleh masuk lagi, agar menahan tangan terlalu lama tidak menggandakan huruf.

Jalankan pengujian tanpa kamera dari folder utama:
    python src/stabilizer.py
"""

from collections import Counter, deque
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config


class PenstabilPrediksi:
    """Menstabilkan prediksi frame-demi-frame menjadi rangkaian teks."""

    def __init__(
        self,
        jumlah_frame=config.JUMLAH_FRAME_VOTING,
        ambang_keyakinan=config.AMBANG_KEYAKINAN,
        detik_stabil=config.DETIK_STABIL,
        detik_jeda_huruf_sama=config.DETIK_JEDA_HURUF_SAMA,
    ):
        self.jumlah_frame = max(1, int(jumlah_frame))
        self.ambang_keyakinan = float(ambang_keyakinan)
        self.detik_stabil = float(detik_stabil)
        self.detik_jeda_huruf_sama = float(detik_jeda_huruf_sama)

        self.riwayat = deque(maxlen=self.jumlah_frame)
        self.huruf_kandidat = None
        self.waktu_mulai_stabil = None
        self.huruf_terakhir_masuk = None
        self.waktu_boleh_sama_lagi = 0.0
        self.teks = ""

    def _hitung_mayoritas(self):
        """Ambil huruf terbanyak di jendela voting. Kembalikan None jika belum cukup suara."""
        suara_valid = [h for h in self.riwayat if h is not None]
        if not suara_valid:
            return None
        # Minimal setengah dari isi jendela saat ini harus suara valid yang sepakat
        batas_suara = max(1, len(self.riwayat) // 2 + 1) if len(self.riwayat) >= 3 else 1
        huruf_terbanyak, jumlah = Counter(suara_valid).most_common(1)[0]
        if jumlah >= batas_suara:
            return huruf_terbanyak
        return None

    def perbarui(self, huruf, keyakinan, waktu=None):
        """Masukkan 1 hasil prediksi frame dan kembalikan status stabilisasi.

        Parameter:
          huruf     : string huruf ('A'-'Z') atau None jika tidak ada tangan
          keyakinan : float 0.0 - 1.0 (skor keyakinan model)
          waktu     : float timestamp detik (opsional, untuk pengujian tanpa kamera)

        Kembalikan dict:
          - huruf_stabil : huruf mayoritas saat ini (atau None)
          - huruf_baru   : huruf yang BARU saja dikonfirmasi masuk ke teks (atau None)
          - progres      : angka 0.0 - 1.0 menuju konfirmasi huruf
          - teks         : string teks yang sudah terkumpul
        """
        if waktu is None:
            waktu = time.monotonic()

        # 1. Saring berdasarkan ambang keyakinan
        if huruf is not None and float(keyakinan) >= self.ambang_keyakinan:
            self.riwayat.append(str(huruf))
        else:
            self.riwayat.append(None)

        # 2. Cari huruf mayoritas dari N frame terakhir
        mayoritas = self._hitung_mayoritas()

        # Jika tidak ada mayoritas (tangan turun / ragu-ragu), reset kandidat
        # dan izinkan huruf terakhir dipakai lagi tanpa harus menunggu jeda penuh.
        if mayoritas is None:
            self.huruf_kandidat = None
            self.waktu_mulai_stabil = None
            self.huruf_terakhir_masuk = None
            return {
                "huruf_stabil": None,
                "huruf_baru": None,
                "progres": 0.0,
                "teks": self.teks,
            }

        # 3. Jika huruf mayoritas berganti ke huruf baru, mulai hitung waktu stabil dari awal
        if mayoritas != self.huruf_kandidat:
            self.huruf_kandidat = mayoritas
            # Jika masih huruf yang sama dengan yang baru saja masuk, hormati waktu jeda
            if mayoritas == self.huruf_terakhir_masuk and waktu < self.waktu_boleh_sama_lagi:
                self.waktu_mulai_stabil = self.waktu_boleh_sama_lagi
            else:
                self.waktu_mulai_stabil = waktu

        # 4. Hitung durasi stabil dan progres (0.0 - 1.0)
        durasi = waktu - self.waktu_mulai_stabil
        if durasi <= 0.0:
            progres = 0.0
        elif self.detik_stabil <= 0.0:
            progres = 1.0
        else:
            progres = min(1.0, durasi / self.detik_stabil)

        # 5. Konfirmasi huruf jika sudah stabil selama detik_stabil
        huruf_baru = None
        if durasi >= self.detik_stabil:
            huruf_baru = mayoritas
            self.teks += huruf_baru
            self.huruf_terakhir_masuk = huruf_baru
            self.waktu_boleh_sama_lagi = waktu + self.detik_jeda_huruf_sama
            # Bersihkan riwayat lama dan mulai hitung ulang setelah masa jeda
            self.riwayat.clear()
            self.huruf_kandidat = None
            self.waktu_mulai_stabil = None
            progres = 0.0

        return {
            "huruf_stabil": mayoritas,
            "huruf_baru": huruf_baru,
            "progres": progres,
            "teks": self.teks,
        }

    def tambah_spasi(self):
        """Tambahkan spasi ke teks."""
        self.teks += " "
        self.huruf_terakhir_masuk = None
        return self.teks

    def hapus_terakhir(self):
        """Hapus 1 karakter terakhir (backspace)."""
        self.teks = self.teks[:-1]
        self.huruf_terakhir_masuk = None
        return self.teks

    def reset(self):
        """Kosongkan teks dan riwayat prediksi."""
        self.riwayat.clear()
        self.huruf_kandidat = None
        self.waktu_mulai_stabil = None
        self.huruf_terakhir_masuk = None
        self.waktu_boleh_sama_lagi = 0.0
        self.teks = ""
        return self.teks


def uji_penstabil():
    """Uji logika PenstabilPrediksi memakai prediksi palsu (tanpa kamera)."""
    print("=== Menguji PenstabilPrediksi tanpa kamera ===")
    penstabil = PenstabilPrediksi(
        jumlah_frame=5,
        ambang_keyakinan=0.7,
        detik_stabil=1.0,
        detik_jeda_huruf_sama=1.0,
    )

    # Skenario 1: Prediksi di bawah ambang keyakinan harus diabaikan
    for i in range(6):
        hasil = penstabil.perbarui("A", 0.55, waktu=i * 0.25)
    assert hasil["teks"] == "", f"Seharusnya kosong, dapat: {hasil['teks']}"
    print("[OK] Skenario 1: Prediksi keyakinan rendah (< 0.7) diabaikan.")

    # Skenario 2: Huruf 'A' stabil 1 detik walaupun ada 1 frame noise 'B' di tengah
    penstabil.reset()
    jadwal_frame = [
        (0.0, "A", 0.90),
        (0.2, "A", 0.88),
        (0.4, "A", 0.92),
        (0.6, "B", 0.85),  # 1 frame noise, tapi 'A' tetap mayoritas di jendela 5 frame
        (0.8, "A", 0.91),
        (1.0, "A", 0.93),  # tepat 1.0 detik sejak t=0.0 -> 'A' harus masuk
    ]
    for t, h, k in jadwal_frame:
        hasil = penstabil.perbarui(h, k, waktu=t)
    assert hasil["teks"] == "A", f"Seharusnya 'A', dapat: {hasil['teks']}"
    assert hasil["huruf_baru"] == "A"
    print("[OK] Skenario 2: Voting tahan terhadap 1 frame noise dan masuk setelah 1 detik.")

    # Skenario 3: Menahan huruf 'A' terus-menerus selama masa jeda tidak langsung menggandakan 'A'
    for t in [1.2, 1.4, 1.6, 1.8]:
        hasil = penstabil.perbarui("A", 0.95, waktu=t)
    assert hasil["teks"] == "A", f"Seharusnya masih 'A' karena jeda, dapat: {hasil['teks']}"
    print("[OK] Skenario 3: Jeda huruf sama mencegah huruf ganda saat tangan ditahan.")

    # Skenario 4: Ganti ke huruf 'B' -> mayoritas berganti ke 'B' pada t=2.4,
    # lalu stabil 1.0 detik pada t=3.4 sehingga teks menjadi 'AB'
    for t in [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4]:
        hasil = penstabil.perbarui("B", 0.92, waktu=t)
    assert hasil["teks"] == "AB", f"Seharusnya 'AB', dapat: {hasil['teks']}"
    print("[OK] Skenario 4: Pergantian huruf ke 'B' berhasil menjadi 'AB'.")

    # Skenario 5: Tombol spasi, hapus, dan reset
    penstabil.tambah_spasi()
    assert penstabil.teks == "AB "
    penstabil.hapus_terakhir()
    assert penstabil.teks == "AB"
    penstabil.hapus_terakhir()
    assert penstabil.teks == "A"
    penstabil.reset()
    assert penstabil.teks == ""
    print("[OK] Skenario 5: Fungsi spasi, hapus (backspace), dan reset bekerja dengan benar.")
    print("\nSemua pengujian stabilizer BERHASIL!")


if __name__ == "__main__":
    uji_penstabil()
