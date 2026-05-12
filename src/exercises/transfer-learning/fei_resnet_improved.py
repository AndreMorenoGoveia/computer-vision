import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, regularizers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

URL = 'http://www.lps.usp.br/hae/apostila/feiCorCrop.zip'
ZIP_NAME = os.path.basename(URL)

if not os.path.exists(ZIP_NAME):
    print(f'Downloading {ZIP_NAME}...')
    os.system(f"wget -nc -U 'Firefox/50.0' {URL}")
if not os.path.exists('todos.csv'):
    os.system(f'unzip -u {ZIP_NAME}')

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

NL, NC = 224, 224


def load_fei(directory, csv_file):
    with open(os.path.join(directory, csv_file)) as f:
        lines = [line.strip().split(';') for line in f]
    n = len(lines)
    x = np.empty((n, NL, NC, 3), dtype='float32')
    y = np.empty(n, dtype='float32')
    for i, (fname, label) in enumerate(lines):
        img = keras_image.load_img(os.path.join(directory, fname),
                                   target_size=(NL, NC))
        arr = keras_image.img_to_array(img)
        x[i] = preprocess_input(np.expand_dims(arr, 0))
        y[i] = float(label)
    return x, y


x, y = load_fei('.', 'todos.csv')
print(f'Dataset loaded: {x.shape}  labels: {y.shape}')

# ---------------------------------------------------------------------------
# TTA helper
# ---------------------------------------------------------------------------

_augmentor = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomTranslation(0.05, 0.05),
])


def predict_tta(model, x_test, n_aug=10):
    """Average predictions over the original image + (n_aug-1) augmented copies."""
    preds = model.predict(x_test, verbose=0)
    for _ in range(n_aug - 1):
        x_aug = _augmentor(x_test, training=True).numpy()
        preds += model.predict(x_aug, verbose=0)
    return preds / n_aug

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

BATCH_SIZE = 10
EPOCHS_P1 = 40
EPOCHS_P2 = 30

acc_per_repete = []

for repete_no in range(1):   # single repetition for speed
    print(f'\n{"=" * 60}\nRepetition {repete_no}\n{"=" * 60}')
    kfold = KFold(n_splits=5, shuffle=False)   # 320 train / 80 test
    ft_tta_accs = []

    for fold_no, (train_idx, test_idx) in enumerate(kfold.split(x, y)):
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_te, y_te = x[test_idx], y[test_idx]
        print(f'\nFold {fold_no}:')

        # ── Build model ──────────────────────────────────────────────────────
        base = ResNet50(weights='imagenet', include_top=False,
                        input_shape=(NL, NC, 3))

        inp = keras.Input(shape=(NL, NC, 3))
        z = layers.RandomFlip('horizontal')(inp)
        z = layers.RandomTranslation(0.05, 0.05)(z)
        z = base(z, training=False)
        z = layers.GlobalAveragePooling2D()(z)
        z = layers.Dropout(0.5)(z)
        z = layers.Dense(256, activation='relu',
                         kernel_regularizer=regularizers.l2(1e-4))(z)
        z = layers.Dropout(0.3)(z)
        out = layers.Dense(1, activation='sigmoid')(z)
        model = Model(inp, out)

        # ── Phase 1: frozen backbone ─────────────────────────────────────────
        base.trainable = False
        model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        ckpt1 = f'fei_hw1_r{repete_no}_f{fold_no}_p1.keras'
        model.fit(
            x_tr, y_tr,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_P1,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                ModelCheckpoint(ckpt1, monitor='val_accuracy',
                                save_best_only=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-6, verbose=0),
            ],
        )
        model = keras.models.load_model(ckpt1)
        sc1 = model.evaluate(x_te, y_te, verbose=0)
        print(f'  Phase 1 (transfer learning): {sc1[1] * 100:.2f}%')

        # ── Phase 2: fine-tune (backbone unfrozen, BN frozen) ────────────────
        loaded_base = model.get_layer('resnet50')
        loaded_base.trainable = True
        for layer in loaded_base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        # Must recompile after changing trainability
        model.compile(
            optimizer=optimizers.Adam(1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        ckpt2 = f'fei_hw1_r{repete_no}_f{fold_no}_p2.keras'
        model.fit(
            x_tr, y_tr,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_P2,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                ModelCheckpoint(ckpt2, monitor='val_accuracy',
                                save_best_only=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-7, verbose=0),
            ],
        )
        best = keras.models.load_model(ckpt2)

        sc2 = best.evaluate(x_te, y_te, verbose=0)
        print(f'  Phase 2 (fine tuning):       {sc2[1] * 100:.2f}%')

        tta_preds = predict_tta(best, x_te, n_aug=10)
        tta_acc = np.mean((tta_preds.flatten() > 0.5) == y_te.astype(bool))
        print(f'  Phase 2 + TTA (10 passes):   {tta_acc * 100:.2f}%')

        ft_tta_accs.append(tta_acc * 100)

    mean_acc = np.mean(ft_tta_accs)
    print(f'\nFold accuracies (TTA): {ft_tta_accs}')
    print(f'Mean accuracy:         {mean_acc:.4f}%')
    acc_per_repete.append(mean_acc)

print(f'\n{"=" * 60}')
print(f'All repetitions: {acc_per_repete}')
print(f'Overall mean:    {np.mean(acc_per_repete):.4f}%')
