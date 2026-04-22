import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from pathlib import Path
from time import time

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Normalization,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "datasets" / "kmnist"


def load_kmnist():
    AX = np.load(DATA_DIR / "kmnist-train-imgs.npz")["arr_0"]
    AY = np.load(DATA_DIR / "kmnist-train-labels.npz")["arr_0"]
    QX = np.load(DATA_DIR / "kmnist-test-imgs.npz")["arr_0"]
    QY = np.load(DATA_DIR / "kmnist-test-labels.npz")["arr_0"]
    return AX, AY, QX, QY


def build_model(normalizer):
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            normalizer,
            Conv2D(32, kernel_size=(5, 5), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(5, 5), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(10, activation=None),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main():
    AX, AY, QX, QY = load_kmnist()
    print(f"AX: {AX.shape}, AY: {AY.shape}, QX: {QX.shape}, QY: {QY.shape}")

    AX = AX.astype("float32")
    QX = QX.astype("float32")
    AX = np.expand_dims(AX, axis=-1)
    QX = np.expand_dims(QX, axis=-1)

    normalizer = Normalization(axis=None)
    normalizer.adapt(AX)

    model = build_model(normalizer)
    model.summary()

    t0 = time()
    model.fit(AX, AY, batch_size=100, epochs=30, verbose=2)
    print(f"Tempo de treino: {time() - t0:.2f} s")

    test_loss, test_accuracy = model.evaluate(QX, QY, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {100 * test_accuracy:.2f} %")
    print(f"Test error: {100 * (1 - test_accuracy):.2f} %")


if __name__ == "__main__":
    main()
