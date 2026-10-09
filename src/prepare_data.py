"""Ekstrak zip dataset, ratakan folder bersarang, buang duplikat/rusak, normalkan gambar.

Langkah per gambar: hapus duplikat (hash isi file), perbaiki rotasi EXIF, ubah ke RGB,
kecilkan jika sisi terpanjang > 1024 px, simpan sebagai JPEG.
"""
import argparse
import hashlib
import io
import os
import shutil
import zipfile

from PIL import Image, ImageOps

CLASSES = ['car', 'motorcycle']
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')


def main(zip_path, out_dir):
    raw = os.path.join(os.path.dirname(out_dir) or '.', 'raw')
    shutil.rmtree(raw, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw)
    for c in CLASSES:
        os.makedirs(os.path.join(out_dir, c))

    seen = set()
    count = {c: 0 for c in CLASSES}
    skipped = {'bukan gambar': 0, 'kelas tidak dikenal': 0, 'duplikat': 0, 'rusak': 0}

    for root, _, files in os.walk(raw):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            if '__MACOSX' in path or fn.startswith('.') or not fn.lower().endswith(EXTS):
                skipped['bukan gambar'] += 1
                continue
            folders = [p.lower() for p in os.path.relpath(path, raw).split(os.sep)[:-1]]
            cls = next((c for c in CLASSES if c in folders), None)
            if cls is None:
                skipped['kelas tidak dikenal'] += 1
                continue
            with open(path, 'rb') as fh:
                data = fh.read()
            h = hashlib.md5(data).hexdigest()
            if h in seen:
                skipped['duplikat'] += 1
                continue
            try:
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
            except Exception:
                skipped['rusak'] += 1
                continue
            seen.add(h)
            im.thumbnail((1024, 1024))
            count[cls] += 1
            im.save(os.path.join(out_dir, cls, f'{cls}_{count[cls]:04d}.jpg'), quality=95)

    print('Gambar bersih per kelas:', count)
    print('Dilewati:', skipped)
    if not all(count.values()):
        raise SystemExit('Ada kelas tanpa gambar. Folder di dalam zip harus memuat "car" dan "motorcycle".')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True, help='path dataset_patrol.zip')
    ap.add_argument('--out', default='data/clean')
    a = ap.parse_args()
    main(a.zip, a.out)
