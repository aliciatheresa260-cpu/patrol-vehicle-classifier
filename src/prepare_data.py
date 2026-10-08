"""Ekstrak zip dataset, flatten folder bersarang, hapus duplikat (hash isi file)."""
import argparse, hashlib, os, shutil, zipfile

CLASSES = ['car', 'motorcycle']
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def main(zip_path, out_dir):
    raw = os.path.join(os.path.dirname(out_dir) or '.', 'raw')
    shutil.rmtree(raw, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw)

    shutil.rmtree(out_dir, ignore_errors=True)
    for c in CLASSES:
        os.makedirs(os.path.join(out_dir, c))

    seen, cnt = set(), {c: 0 for c in CLASSES}
    for root, _, files in os.walk(raw):
        for f in files:
            if not f.lower().endswith(EXTS):
                continue
            p = os.path.join(root, f)
            parts = [x.lower() for x in p.split(os.sep)]
            c = next((c for c in CLASSES if c in parts), None)
            if c is None:
                continue
            with open(p, 'rb') as fh:
                h = hashlib.md5(fh.read()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            cnt[c] += 1
            ext = os.path.splitext(f)[1].lower()
            shutil.copy(p, os.path.join(out_dir, c, f'{c}_{cnt[c]:04d}{ext}'))
    print('Gambar unik per kelas:', cnt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True, help='path dataset_patrol.zip')
    ap.add_argument('--out', default='data/clean')
    a = ap.parse_args()
    main(a.zip, a.out)
