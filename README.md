# Patrol Vehicle Classifier

Klasifikasi gambar **mobil (car)** vs **motor (motorcycle)** memakai transfer learning
**MobileNetV3Large** (TensorFlow/Keras).

## Hasil
- Dataset: 114 gambar unik (54 car, 60 motorcycle) setelah deduplikasi
- Split stratified 70/15/15 (train/val/test)
- Test accuracy: 18/18 benar (100%)

> Catatan: test set hanya 18 gambar dan beberapa gambar mungkin berasal dari frame video
> yang berdekatan, sehingga skor bisa terlalu optimistis. Uji dengan foto baru dari kondisi
> nyata (malam, hujan, sudut miring, tertutup sebagian) sebelum dipakai sungguhan.

## Struktur
```
src/prepare_data.py   # ekstrak zip, flatten, hapus duplikat
src/train.py          # training + fine-tuning + evaluasi test set
src/predict.py        # prediksi gambar baru
data/                 # dataset (tidak di-commit)
models/               # model hasil training (tidak di-commit)
```

## Cara pakai
```bash
pip install -r requirements.txt

python src/prepare_data.py --zip dataset_patrol.zip --out data/clean
python src/train.py --data data/clean
python src/predict.py foto1.jpg foto2.jpg --model models/final.keras
```

## Perbaikan dari versi awal (akurasi 27%)
| Masalah | Perbaikan |
|---|---|
| `CategoricalCrossentropy` dengan label integer | `SparseCategoricalCrossentropy` konsisten di semua tahap |
| `Rescaling(1/255)` + `include_preprocessing=False` | `include_preprocessing=True`, input 0..255 |
| Split acak tanpa test set | Split stratified + evaluasi di test set |
| Augmentasi lemah | Flip, rotasi, zoom, translasi, kontras, brightness |
| Fine-tuning ikut melatih BatchNorm | BatchNorm dibekukan, LR 2e-5, `ReduceLROnPlateau` |

## Lisensi
Tambahkan sesuai kebutuhan (mis. MIT).
