import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential


CLASS_NAMES = [
    "Camiseta",
    "Calca",
    "Pulover",
    "Vestido",
    "Casaco",
    "Sandalia",
    "Camisa",
    "Tenis",
    "Bolsa",
    "Botins",
]


def build_model():
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ]
    )

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_predictions(images, true_labels, predicted_labels):
    fig = plt.figure(figsize=(10, 10))

    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
        ax.set_title(CLASS_NAMES[true_labels[i]], color="b", fontsize=10, pad=2)
        ax.text(
            0.5,
            -0.12,
            CLASS_NAMES[predicted_labels[i]],
            color="r",
            fontsize=10,
            ha="center",
            va="top",
            transform=ax.transAxes,
        )

    fig.suptitle("Fashion-MNIST: azul = verdadeiro, vermelho = previsto", fontsize=12)
    fig.tight_layout()
    plt.show()


def main():
    (AX, AY), (QX, QY) = fashion_mnist.load_data()

    AX = AX.astype("float32") / 255.0
    QX = QX.astype("float32") / 255.0
    AX = np.expand_dims(AX, axis=-1)
    QX = np.expand_dims(QX, axis=-1)

    model = build_model()
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
    )
    model.fit(
        AX,
        AY,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(QX, QY, verbose=0)
    error_rate = 1.0 - test_accuracy

    predictions = model.predict(QX, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    print(f"Acuracia no conjunto de teste: {test_accuracy * 100:.2f}%")
    print(f"Taxa de erro no conjunto de teste: {error_rate * 100:.2f}%")
    print(f"Loss no conjunto de teste: {test_loss:.4f}")

    plot_predictions(QX.squeeze(-1), QY, predicted_labels)


if __name__ == "__main__":
    main()
