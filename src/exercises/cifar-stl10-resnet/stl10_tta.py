import os
import gdown
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# --- Load STL-10 test data ---

file_id = '1vAVoVbz24HDU98RiorptkvpLTX4qp_VD'
output = 'stl10_labeled.npz'
if not os.path.exists(output):
    gdown.download(
        f'https://drive.google.com/file/d/{file_id}/view?usp=drive_link',
        output, quiet=False, fuzzy=True
    )

with np.load(output) as data:
    X_test = data['X_test']
    y_test = data['y_test']

X_test = np.rot90(X_test, k=-1, axes=(1, 2))
y_test = y_test - 1
X_test = 2 * (X_test / 255.0 - 0.5)

print(f'Test set: {X_test.shape}')

# --- Load saved model from homework 1 ---
# Expects stl10_resnet_aug.keras to exist (produced by homework1_stl10_resnet.py)

model_path = 'stl10_resnet_aug.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f'Model file "{model_path}" not found. '
        'Run homework1_stl10_resnet.py first to train and save the model.'
    )

model = keras.models.load_model(model_path)
print('Model loaded.')

# --- Baseline accuracy (no TTA) ---

preds_baseline = model.predict(X_test, batch_size=64, verbose=0)
acc_baseline = np.mean(np.argmax(preds_baseline, axis=1) == y_test)
print(f'Baseline accuracy (no TTA): {acc_baseline:.4f}')

# --- Test-Time Augmentation ---
# Preprocessing layers are training-only within a model, so we build a
# separate small network that applies random transforms at call time.

# The augmentation network is called explicitly with training=True so that
# random layers fire during inference.
input_shape = X_test.shape[1:]  # (96, 96, 3)

aug_input = keras.Input(shape=input_shape)
aug_x = layers.RandomFlip('horizontal')(aug_input)
aug_x = layers.RandomRotation(0.1)(aug_x)
aug_x = layers.RandomTranslation(0.1, 0.1)(aug_x)
augmentor = keras.Model(aug_input, aug_x)

num_augmentations = 20
all_preds = []

print(f'Running TTA with {num_augmentations} augmented versions per image...')
for i in range(num_augmentations):
    # training=True forces random layers to sample new transforms each call
    distorted = augmentor(X_test, training=True).numpy()
    preds = model.predict(distorted, batch_size=64, verbose=0)
    all_preds.append(preds)
    if (i + 1) % 5 == 0:
        print(f'  {i + 1}/{num_augmentations} done')

avg_preds = np.mean(all_preds, axis=0)
acc_tta = np.mean(np.argmax(avg_preds, axis=1) == y_test)
print(f'\nTTA accuracy ({num_augmentations} augmentations): {acc_tta:.4f}')
print(f'Improvement over baseline: {acc_tta - acc_baseline:+.4f}')
