"""Objetivo 2 - Modelo de maior acuracia possivel.

Estrategia: ViT maior (DeiT3-base distilled patch16 224) com transfer
learning em duas fases e data augmentation mais forte. ViTs grandes
costumam superar EfficientNetB0 e DeiT-Tiny no Oxford Flowers 102 e
mantem boa robustez a oclusoes distribuidas.
"""

from __future__ import annotations

import json
from time import time

import tensorflow as tf
from tensorflow.keras import layers, optimizers

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
from train_vit import vit_preprocess

silence_tf_logs()
gpu_memory_growth()

BEST_PRESET = "vit_base_patch16_224_imagenet"


def build_model() -> tf.keras.Model:
    import keras_hub

    backbone = keras_hub.models.ViTBackbone.from_preset(BEST_PRESET)
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(vit_preprocess, name="preprocess")(inputs)
    features = backbone(x)
    if isinstance(features, dict):
        features = features.get("sequence_output", features.get("pooled_output"))
    if features.shape.rank == 3:
        cls = features[:, 0]
    else:
        cls = features
    cls = layers.Dropout(0.1)(cls)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(cls)
    return tf.keras.Model(inputs, outputs, name="vit_base_flowers"), backbone


def main(
    head_epochs: int = 15,
    fine_tune_epochs: int = 25,
    batch_size: int = 16,
) -> None:
    train_ds, val_ds, (test_x, test_y), _ = load_oxford_flowers(batch_size=batch_size)

    model, backbone = build_model()
    model.compile(
        optimizer=optimizers.AdamW(1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = MODELS_DIR / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True
        ),
    ]

    t0 = time()
    print("\n=== Fase 1: head-only ===")
    model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=callbacks, verbose=2)

    print("\n=== Fase 2: fine-tuning ===")
    backbone.trainable = True
    model.compile(
        optimizer=optimizers.AdamW(1e-5, weight_decay=1e-4),
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
    print(f"\nTempo total de treino best: {train_time:.1f} s")

    model = tf.keras.models.load_model(ckpt_path)
    accuracy = evaluate_accuracy(model, test_x, test_y)
    print(f"Acuracia teste (sem oclusao): {accuracy:.4f}")

    metrics = {
        "model": BEST_PRESET,
        "train_time_seconds": train_time,
        "test_accuracy_clean": accuracy,
    }
    with open(RESULTS_DIR / "best_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
