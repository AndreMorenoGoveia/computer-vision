import os
import gdown
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

# --- Data Loading ---

file_id = '1vAVoVbz24HDU98RiorptkvpLTX4qp_VD'
output = 'stl10_labeled.npz'
if not os.path.exists(output):
    gdown.download(
        f'https://drive.google.com/file/d/{file_id}/view?usp=drive_link',
        output, quiet=False, fuzzy=True
    )

with np.load(output) as data:
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']

# STL-10 images are stored transposed — rotate back to correct orientation
X_train = np.rot90(X_train, k=-1, axes=(1, 2))
X_test  = np.rot90(X_test,  k=-1, axes=(1, 2))

# Labels are 1-indexed in STL-10
y_train = y_train - 1
y_test  = y_test  - 1

# Normalize to [-1, 1]
X_train = 2 * (X_train / 255.0 - 0.5)
X_test  = 2 * (X_test  / 255.0 - 0.5)

num_classes = 10
input_shape = X_train.shape[1:]  # (96, 96, 3)

print(f'Train: {X_train.shape}, Test: {X_test.shape}')
print(f'Label range: {y_train.min()} - {y_train.max()}')


# --- ResNet Building Blocks ---

def resnet_layer(inputs, num_filters=64, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True):
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                      padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(inputs)
    if batch_normalization:
        x = layers.BatchNormalization()(x)
    if activation is not None:
        x = layers.Activation(activation)(x)
    return x


def build_resnet(input_shape, num_classes, num_filters=64, num_blocks=3, augment=False):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    if augment:
        x = layers.RandomFlip('horizontal')(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomTranslation(0.1, 0.1)(x)

    # Initial conv to reduce spatial size: 96x96 -> 48x48
    x = resnet_layer(x, num_filters=num_filters, kernel_size=3, strides=2)

    # Stack of residual blocks
    for block in range(num_blocks):
        # Double filters and halve spatial every block except the first
        if block > 0:
            strides = 2
            num_filters *= 2
        else:
            strides = 1

        # Residual path
        y = resnet_layer(x, num_filters=num_filters, strides=strides)
        y = resnet_layer(y, num_filters=num_filters, activation=None)

        # Projection shortcut when shape changes
        if strides > 1 or x.shape[-1] != num_filters:
            x = resnet_layer(x, num_filters=num_filters, kernel_size=1,
                             strides=strides, activation=None, batch_normalization=False)

        x = layers.add([x, y])
        x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_regularizer=l2(1e-4))(x)

    return keras.Model(inputs, outputs)


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


def train_model(augment, save_path, epochs=200):
    model = build_resnet(input_shape, num_classes, num_filters=64,
                         num_blocks=3, augment=augment)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        LearningRateScheduler(lr_schedule),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True,
                        verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2
    )

    best_model = keras.models.load_model(save_path)
    _, acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f'\n[{"WITH" if augment else "WITHOUT"} augmentation] Best test accuracy: {acc:.4f}')
    return history


# --- Run ---

print('\n=== Training WITHOUT data augmentation ===')
train_model(augment=False, save_path='stl10_resnet_no_aug.keras', epochs=200)

print('\n=== Training WITH data augmentation ===')
train_model(augment=True, save_path='stl10_resnet_aug.keras', epochs=200)
