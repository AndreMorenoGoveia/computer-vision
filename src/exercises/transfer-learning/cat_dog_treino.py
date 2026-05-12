import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, regularizers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

URL = 'http://www.lps.usp.br/hae/psi3472/ep-2021/cat_dog_clean.zip'
ZIP_NAME = os.path.basename(URL)

if not os.path.exists(ZIP_NAME):
    print(f'Downloading {ZIP_NAME}...')
    os.system(f"wget -nc -U 'Firefox/50.0' {URL}")
if not os.path.exists('cat_dog_clean'):
    os.system(f'unzip -u {ZIP_NAME}')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAIN_DIR = os.path.join('cat_dog_clean', 'dog-and-cat', 'training_set')
print(f'Training directory: {TRAIN_DIR}')

# ---------------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------------

NL, NC = 224, 224
BATCH_SIZE = 32
VAL_SPLIT = 0.1
SEED = 42


def make_dataset(subset):
    return tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(NL, NC),
        batch_size=BATCH_SIZE,
        label_mode='binary',   # cats=0, dogs=1
        shuffle=True,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset=subset,
    )


def apply_preprocess(images, labels):
    return preprocess_input(images), labels


raw_train_ds = make_dataset('training')
raw_val_ds = make_dataset('validation')

class_names = raw_train_ds.class_names
print(f'Class names (label 0 / 1): {class_names}')

train_ds = raw_train_ds.map(apply_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = raw_val_ds.map(apply_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

base = ResNet50(weights='imagenet', include_top=False, input_shape=(NL, NC, 3))

inp = keras.Input(shape=(NL, NC, 3))
z = layers.RandomFlip('horizontal')(inp)
z = layers.RandomTranslation(0.1, 0.1)(z)
z = base(z, training=False)
z = layers.GlobalAveragePooling2D()(z)
z = layers.Dropout(0.5)(z)
z = layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4))(z)
z = layers.Dropout(0.3)(z)
out = layers.Dense(1, activation='sigmoid')(z)
model = Model(inp, out)

# ---------------------------------------------------------------------------
# Phase 1: frozen backbone
# ---------------------------------------------------------------------------

base.trainable = False
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)
model.summary()

print('\nPhase 1: transfer learning (backbone frozen) …')
model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[
        ModelCheckpoint('transf_p1.keras', monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ],
)
model = keras.models.load_model('transf_p1.keras')
print('Phase 1 done — best checkpoint loaded.')

# ---------------------------------------------------------------------------
# Phase 2: fine-tune (backbone unfrozen, BN layers frozen)
# ---------------------------------------------------------------------------

loaded_base = model.get_layer('resnet50')
loaded_base.trainable = True
for layer in loaded_base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

print('\nPhase 2: fine-tuning (all layers, BN frozen) …')
model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    callbacks=[
        ModelCheckpoint('transf.keras', monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ],
)

print('\nBest model saved as transf.keras')
print('Run cat_dog_teste.py to evaluate on the test set.')
