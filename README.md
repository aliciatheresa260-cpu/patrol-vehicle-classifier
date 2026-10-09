# Patrol Vehicle Classifier

Klasifikasi gambar **mobil (car)** vs **motor (motorcycle)** memakai transfer learning
**MobileNetV3-Large** (TensorFlow/Keras), dibuat untuk robot patroli.

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliciatheresa260-cpu/patrol-vehicle-classifier/blob/main/notebooks/train_colab.ipynb)

## Hasil

Dataset: 114 gambar unik (54 car, 60 motorcycle), tanpa duplikat. Split stratified 70/15/15
(79 train / 17 val / 18 test), seed 42.

| Data | Akurasi |
|---|---|
| Test (18, tidak dipakai training) | 18/18 (100%) |
| Val (17) | 17/17 (100%) |
| Train (79) | 79/79 (100%) |

Uji ketahanan pada 35 gambar yang tidak dipakai melatih (val + test), setelah gambar sengaja dirusak:

| Kondisi | Akurasi |
|---|---|
| Asli | 100% |
| Gelap (25% terang) | 100% |
| Miring 30° | 100% |
| Noise | 100% |
| Hitam-putih | 100% |
| Resolusi rendah (48 px) | 97,1% |
| Buram | 97,1% |
| Objek terpotong (crop tengah 50%) | 91,4% |

**Catatan jujur:**
- Test set hanya 18 gambar. Benar 18/18 hanya menjamin akurasi sebenarnya di atas sekitar 82% (batas bawah interval kepercayaan 95%), belum membuktikan 100%.
- Uji ketahanan memakai gambar dari dataset yang sama, bukan foto dari kamera robot. Uji dengan foto baru dari kondisi nyata (malam, hujan, sudut miring, tertutup sebagian) sebelum dipakai sungguhan.
- Untuk angka yang lebih stabil, jalankan 5-fold cross-validation di notebook (`RUN_CV = True`).
- Model ini mengklasifikasi **satu label per gambar**. Kalau robot perlu menemukan kendaraan di dalam adegan yang ramai, gunakan model deteksi (mis. YOLO atau SSD-MobileNet).

## Struktur
```
notebooks/train_colab.ipynb   # notebook Colab lengkap (upload zip -> latih -> evaluasi -> simpan -> prediksi)
src/prepare_data.py           # ekstrak zip, ratakan folder, hapus duplikat/rusak
src/train.py                  # training 2 tahap + evaluasi test set
src/predict.py                # prediksi gambar baru
data/                         # dataset (tidak di-commit)
models/                       # model hasil training (tidak di-commit)
```

## Cara pakai

**Di Google Colab (disarankan):** klik tombol "Buka di Colab" di atas, pilih Runtime → T4 GPU,
lalu Run all dan upload `dataset_patrol.zip` saat diminta.

**Di komputer sendiri:**
```bash
pip install -r requirements.txt

python src/prepare_data.py --zip dataset_patrol.zip --out data/clean
python src/train.py --data data/clean
python src/predict.py foto1.jpg foto2.jpg --model models/final.keras
```

Bobot ImageNet diunduh otomatis oleh Keras saat pertama kali model dibuat (perlu internet).

## Aturan penting preprocessing
- Input model adalah piksel **0–255**. Preprocessing MobileNetV3 sudah ada di dalam model
  (`include_preprocessing=True`), jadi **jangan** menambahkan `Rescaling(1/255)`.
- Saat prediksi, gunakan resize dengan padding yang sama seperti saat training (`src/predict.py` sudah melakukannya).
- Untuk TFLite (Raspberry Pi/Jetson), input juga tetap float32 0–255.

## Perbaikan dari versi awal (akurasi 27%)
| Masalah | Perbaikan |
|---|---|
| `CategoricalCrossentropy` dengan label integer | `SparseCategoricalCrossentropy` konsisten di semua tahap |
| `Rescaling(1/255)` + `include_preprocessing=False` | `include_preprocessing=True`, input 0..255 |
| Split acak tanpa test set | Split stratified 70/15/15 + evaluasi di test set |
| Augmentasi lemah | Flip, rotasi, zoom, translasi, kontras, brightness |
| Fine-tuning ikut melatih BatchNorm | BatchNorm dibekukan, LR 2e-5, `ReduceLROnPlateau` |
| Gambar dipaksa persegi (rasio penyok) | `resize_with_pad` |

## Lisensi
Tambahkan sesuai kebutuhan (mis. MIT).
