import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from pathlib import Path
from time import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model, Sequential


DIGITS = (1, 4)
RESULTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "results" / "mnist_1_4"


def filter_digits(X, Y, digits):
    mask = np.isin(Y, digits)
    X = X[mask]
    Y = Y[mask]
    remap = {d: i for i, d in enumerate(digits)}
    Y = np.array([remap[y] for y in Y], dtype=np.int64)
    return X, Y


def build_model():
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(20, kernel_size=(5, 5), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(40, kernel_size=(5, 5), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(100, activation="relu"),
            Dense(len(DIGITS), activation=None),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def plot_filters(model, path):
    filters, _ = model.get_layer(index=0).get_weights()
    filters = np.squeeze(filters)
    n_filters = filters.shape[2]
    stacked = np.stack([filters[:, :, i] for i in range(n_filters)], axis=0)
    vmax = float(np.max(np.abs(stacked)))

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(6, 5))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(stacked[i], vmin=-vmax, vmax=vmax, cmap="gray")
        ax.axis("off")
    fig.subplots_adjust(right=0.82)
    cax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cax)
    fig.suptitle(f"Filtros 5x5 da 1a camada (distinguir {DIGITS[0]} vs {DIGITS[1]})")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_activations(model, image, digit, path):
    activation_model = Model(inputs=model.inputs, outputs=model.get_layer(index=0).output)
    x = np.expand_dims(image, axis=(0, -1)).astype("float32")
    activations = activation_model.predict(x, verbose=0)
    activations = np.squeeze(activations, axis=0)
    n_filters = activations.shape[-1]
    stacked = np.stack([activations[:, :, i] for i in range(n_filters)], axis=0)
    vmax = float(np.max(stacked)) or 1.0

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(6, 5))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(stacked[i], vmin=0, vmax=vmax, cmap="gray")
        ax.axis("off")
    fig.subplots_adjust(right=0.82)
    cax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cax)
    fig.suptitle(f"Ativacoes da 1a camada para digito {digit}")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_digit(image, digit, path):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    ax.set_title(f"Digito {digit}")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (AX, AY), (QX, QY) = mnist.load_data()
    AX, AY = filter_digits(AX, AY, DIGITS)
    QX, QY = filter_digits(QX, QY, DIGITS)
    print(f"Treino: {AX.shape}, Teste: {QX.shape}")

    AX = (AX.astype("float32") / 255.0) - 0.5
    QX = (QX.astype("float32") / 255.0) - 0.5
    AX = np.expand_dims(AX, axis=-1)
    QX_cnn = np.expand_dims(QX, axis=-1)

    model = build_model()
    model.summary()

    t0 = time()
    model.fit(AX, AY, batch_size=100, epochs=10, verbose=2)
    print(f"Tempo de treino: {time() - t0:.2f} s")

    test_loss, test_accuracy = model.evaluate(QX_cnn, QY, verbose=0)
    print(f"Test accuracy: {100 * test_accuracy:.2f} %")
    print(f"Test error: {100 * (1 - test_accuracy):.2f} %")

    plot_filters(model, RESULTS_DIR / "filtros_1a_camada.png")
    print(f"Filtros salvos em {RESULTS_DIR / 'filtros_1a_camada.png'}")

    for class_idx, digit in enumerate(DIGITS):
        idx = int(np.where(QY == class_idx)[0][0])
        image = QX[idx]
        save_digit(image, digit, RESULTS_DIR / f"exemplo_{digit}.png")
        plot_activations(
            model,
            image,
            digit,
            RESULTS_DIR / f"ativacoes_{digit}.png",
        )
        print(f"Ativacoes para digito {digit} salvas.")


if __name__ == "__main__":
    main()
