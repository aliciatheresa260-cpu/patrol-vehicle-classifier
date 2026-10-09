"""Training MobileNetV3Large (transfer learning + fine-tuning) untuk car vs motorcycle.

Input model = piksel 0..255 (preprocessing MobileNetV3 sudah ada di dalam model).
Jangan menambahkan Rescaling(1/255).
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large

CLASSES = ['car', 'motorcycle']
IMG = 224


def list_files(data_dir):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        for f in sorted(os.listdir(os.path.join(data_dir, c))):
            paths.append(os.path.join(data_dir, c, f))
            labels.append(i)
    return np.array(paths), np.array(labels)


def load_image(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize_with_pad(img, IMG, IMG)   # rasio gambar dijaga, tetap 0..255
    img.set_shape((IMG, IMG, 3))
    return tf.cast(img, tf.float32)


def make_ds(p, y, batch, train, seed):
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    ds = ds.map(lambda path, label: (load_image(path), label),
                num_parallel_calls=tf.data.AUTOTUNE).cache()
    if train:
        ds = ds.shuffle(len(p), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(weights):
    augment = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.15),
        layers.RandomZoom((-0.2, 0.2)),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name='augmentation')
    base = MobileNetV3Large(input_shape=(IMG, IMG, 3), include_top=False,
                            weights=weights, include_preprocessing=True)
    base.trainable = False
    inp = tf.keras.Input((IMG, IMG, 3))
    x = augment(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation='softmax')(x)
    return models.Model(inp, out), base


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping('val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.3, patience=3, min_lr=1e-7),
    ]


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])


def main(a):
    tf.keras.utils.set_random_seed(a.seed)
    os.makedirs(a.models_dir, exist_ok=True)

    paths, labels = list_files(a.data)
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=a.seed)
    p_val, p_te, y_val, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=a.seed)
    print(f'train={len(p_tr)} val={len(p_val)} test={len(p_te)}')

    train_ds = make_ds(p_tr, y_tr, a.batch, True, a.seed)
    val_ds = make_ds(p_val, y_val, a.batch, False, a.seed)
    test_ds = make_ds(p_te, y_te, a.batch, False, a.seed)

    cw = {i: len(y_tr) / (len(CLASSES) * n) for i, n in enumerate(np.bincount(y_tr, minlength=len(CLASSES)))}
    weights = None if a.weights == 'none' else a.weights
    model, base = build_model(weights)

    # Tahap 1: latih head saja
    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, class_weight=cw, callbacks=callbacks())

    # Tahap 2: fine-tuning N layer terakhir, BatchNorm tetap beku
    base.trainable = True
    for layer in base.layers[:-a.unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 2e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=a.ft_epochs, class_weight=cw, callbacks=callbacks())

    # Evaluasi di test set
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f'Test loss={loss:.4f} acc={acc * 100:.1f}%')
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    print(classification_report(y_te, y_pred, target_names=CLASSES, zero_division=0))
    print(confusion_matrix(y_te, y_pred, labels=range(len(CLASSES))))

    model.save(os.path.join(a.models_dir, 'final.keras'))
    with open(os.path.join(a.models_dir, 'classes.json'), 'w') as f:
        json.dump(CLASSES, f)
    print('Model tersimpan di', a.models_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/clean')
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=30, help='epoch tahap 1 (head)')
    ap.add_argument('--ft-epochs', type=int, default=40, help='epoch tahap 2 (fine-tuning)')
    ap.add_argument('--unfreeze', type=int, default=30, help='jumlah layer terakhir yang dilatih saat fine-tuning')
    ap.add_argument('--weights', default='imagenet', help="'imagenet' (default) atau 'none' (hanya untuk uji cepat)")
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args())
