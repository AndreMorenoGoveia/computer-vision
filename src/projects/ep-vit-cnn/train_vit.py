"""Tarefa 1 - Baseline ViT (DeiT-Tiny distilled) no Oxford Flowers 102.

Usa backbone do keras_hub. Estrategia em duas fases (head-only +
fine-tuning) para alinhar com o treino da CNN.
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

silence_tf_logs()
gpu_memory_growth()

VIT_PRESET = "deit_tiny_distilled_patch16_224_imagenet"

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406]) * 255.0
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225]) * 255.0


def vit_preprocess(x: tf.Tensor) -> tf.Tensor:
    """Normaliza para o padrao ImageNet usado pelo DeiT."""
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def build_model() -> tf.keras.Model:
    import keras_hub

    backbone = keras_hub.models.DeiTBackbone.from_preset(VIT_PRESET)
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
    return tf.keras.Model(inputs, outputs, name="deit_tiny_flowers"), backbone


def main(
    head_epochs: int = 15,
    fine_tune_epochs: int = 20,
    batch_size: int = 32,
) -> None:
    train_ds, val_ds, (test_x, test_y), _ = load_oxford_flowers(batch_size=batch_size)

    model, backbone = build_model()
    model.compile(
        optimizer=optimizers.AdamW(1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = MODELS_DIR / "vit_best.keras"
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
    print(f"\nTempo total de treino ViT: {train_time:.1f} s")

    model = tf.keras.models.load_model(ckpt_path)
    accuracy = evaluate_accuracy(model, test_x, test_y)
    print(f"Acuracia teste (sem oclusao): {accuracy:.4f}")

    metrics = {
        "model": VIT_PRESET,
        "train_time_seconds": train_time,
        "test_accuracy_clean": accuracy,
    }
    with open(RESULTS_DIR / "vit_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
