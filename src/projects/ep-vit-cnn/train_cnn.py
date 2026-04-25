"""Tarefa 1 - Baseline CNN (EfficientNetB0) no Oxford Flowers 102.

Estrategia em duas fases:
    1. Head-only: backbone congelado, treina apenas a cabeca densa.
    2. Fine-tuning: descongela o backbone com learning rate baixo.
       BatchNormalization permanece nao-treinavel (recomendacao do enunciado).
"""

from __future__ import annotations

import json
from time import time

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efn_preprocess

from common import (
    IMG_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
    RESULTS_DIR,
    evaluate_accuracy,
    gpu_memory_growth,
    load_oxford_flowers,
    silence_tf_logs,
)

silence_tf_logs()
gpu_memory_growth()


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(efn_preprocess, name="preprocess")(inputs)
    backbone = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="efficientnetb0_flowers")


def freeze_batchnorm(model: tf.keras.Model) -> None:
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def main(
    head_epochs: int = 10,
    fine_tune_epochs: int = 20,
    batch_size: int = 32,
) -> None:
    train_ds, val_ds, (test_x, test_y), _ = load_oxford_flowers(batch_size=batch_size)

    model = build_model()
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = MODELS_DIR / "cnn_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
        ),
    ]

    t0 = time()
    print("\n=== Fase 1: head-only ===")
    model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=callbacks, verbose=2)

    print("\n=== Fase 2: fine-tuning ===")
    backbone = model.get_layer(index=2)
    if hasattr(backbone, "trainable"):
        backbone.trainable = True
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    freeze_batchnorm(model)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time() - t0
    print(f"\nTempo total de treino CNN: {train_time:.1f} s")

    model = tf.keras.models.load_model(ckpt_path)

    accuracy = evaluate_accuracy(model, test_x, test_y)
    print(f"Acuracia teste (sem oclusao): {accuracy:.4f}")

    metrics = {
        "model": "EfficientNetB0",
        "train_time_seconds": train_time,
        "test_accuracy_clean": accuracy,
    }
    with open(RESULTS_DIR / "cnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
