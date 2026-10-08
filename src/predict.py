"""Prediksi satu atau banyak gambar dengan model yang sudah dilatih."""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img

CLASSES = ['car', 'motorcycle']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--model', default='models/final.keras')
    a = ap.parse_args()
    model = tf.keras.models.load_model(a.model)
    for path in a.images:
        x = np.expand_dims(img_to_array(load_img(path, target_size=(224, 224))), 0)  # 0..255
        p = model.predict(x, verbose=0)[0]
        print(f'{path}: {CLASSES[p.argmax()]} ({p.max() * 100:.1f}%)')
