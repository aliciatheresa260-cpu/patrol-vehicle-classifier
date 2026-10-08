"""Training MobileNetV3Large (transfer learning + fine-tuning) untuk car vs motorcycle."""
import argparse, os, random
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


def make_ds(p, y, batch, train, seed):
    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))  # tetap 0..255
        img.set_shape((IMG, IMG, 3))
        return img, label
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if train:
        ds = ds.shuffle(len(p), seed=seed, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model():
    augment = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.15),
        layers.RandomZoom((-0.2, 0.2)),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ])
    # include_preprocessing=True -> input 0..255, jangan tambah Rescaling(1/255)
    base = MobileNetV3Large(input_shape=(IMG, IMG, 3), include_top=False,
                            weights='imagenet', include_preprocessing=True)
    base.trainable = False
    inp = tf.keras.Input((IMG, IMG, 3))
    x = augment(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation='softmax')(x)
    return models.Model(inp, out), base


def callbacks(path):
    return [
        tf.keras.callbacks.EarlyStopping('val_loss', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.3, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(path, monitor='val_loss', save_best_only=True),
    ]


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)
    os.makedirs(a.models_dir, exist_ok=True)

    paths, labels = list_files(a.data)
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=a.seed)
    p_val, p_te, y_val, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=a.seed)
    print(f'train={len(p_tr)} val={len(p_val)} test={len(p_te)}')

    train_ds = make_ds(p_tr, y_tr, a.batch, True, a.seed)
    val_ds = make_ds(p_val, y_val, a.batch, False, a.seed)
    test_ds = make_ds(p_te, y_te, a.batch, False, a.seed)

    cw = {i: len(y_tr) / (2 * n) for i, n in enumerate(np.bincount(y_tr))}
    model, base = build_model()

    # Tahap 1: latih head saja
    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, class_weight=cw,
              callbacks=callbacks(os.path.join(a.models_dir, 'stage1.keras')))

    # Tahap 2: fine-tuning 30 layer terakhir, BatchNorm tetap beku
    base.trainable = True
    for l in base.layers[:-30]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    compile_model(model, 2e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=a.ft_epochs, class_weight=cw,
              callbacks=callbacks(os.path.join(a.models_dir, 'best.keras')))

    # Evaluasi di test set
    print('Test loss/acc:', model.evaluate(test_ds, verbose=0))
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    print(classification_report(y_te, y_pred, target_names=CLASSES))
    print(confusion_matrix(y_te, y_pred))

    model.save(os.path.join(a.models_dir, 'final.keras'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/clean')
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--ft-epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args())
