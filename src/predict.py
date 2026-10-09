"""Prediksi satu atau banyak gambar dengan model yang sudah dilatih.

Preprocessing sama persis dengan saat training: RGB, resize dengan padding ke 224x224,
nilai piksel 0..255 (tanpa dibagi 255).
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMG = 224
DEFAULT_CLASSES = ['car', 'motorcycle']


def load_classes(model_path):
    p = os.path.join(os.path.dirname(model_path), 'classes.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return DEFAULT_CLASSES


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--model', default='models/final.keras')
    a = ap.parse_args()

    classes = load_classes(a.model)
    model = tf.keras.models.load_model(a.model)
    for path in a.images:
        im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        x = tf.image.resize_with_pad(np.array(im), IMG, IMG)
        p = model.predict(tf.expand_dims(x, 0), verbose=0)[0]
        print(f'{path}: {classes[int(p.argmax())]} ({p.max() * 100:.1f}%)')
