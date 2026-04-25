"""Utilitarios compartilhados para o EP ViT vs CNN (Oxford Flowers 102).

Concentra carregamento do dataset, redimensionamento com padding,
funcoes de oclusao (central e distribuida) e plots auxiliares.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

IMG_SIZE = 224
NUM_CLASSES = 102
TEST_PER_CLASS = 20
PATCH_SIZE = 16
OCCLUSION_RATIO = 0.25
CENTRAL_OCCLUSION_SIZE = 112

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent.parent.parent / "artifacts"
MODELS_DIR = ARTIFACTS / "models" / "ep-vit-cnn"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE


def resize_with_pad(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def load_oxford_flowers(batch_size: int = 32, augment: bool = True):
    """Carrega Oxford Flowers 102 ja redimensionado.

    Retorna (train_ds, val_ds, test_ds) onde test_ds usa apenas as 20
    primeiras imagens de cada classe (102 classes -> 2040 imagens).
    """
    import tensorflow_datasets as tfds

    (train_raw, val_raw, test_raw), info = tfds.load(
        "oxford_flowers102",
        split=["train", "validation", "test"],
        with_info=True,
        as_supervised=True,
    )

    train_ds = train_raw.map(resize_with_pad, num_parallel_calls=AUTOTUNE)
    val_ds = val_raw.map(resize_with_pad, num_parallel_calls=AUTOTUNE)

    test_arrays = build_balanced_test_set(test_raw)

    if augment:
        augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ]
        )
        train_ds = train_ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_arrays, info


def build_balanced_test_set(test_raw) -> Tuple[np.ndarray, np.ndarray]:
    """Pega as primeiras TEST_PER_CLASS imagens de cada classe.

    Retorna numpy arrays para facilitar oclusao posterior.
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.int32)
    images, labels = [], []
    for image, label in test_raw:
        c = int(label.numpy())
        if counts[c] < TEST_PER_CLASS:
            img = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE).numpy().astype(np.float32)
            images.append(img)
            labels.append(c)
            counts[c] += 1
        if int(counts.sum()) >= NUM_CLASSES * TEST_PER_CLASS:
            break
    return np.stack(images), np.array(labels, dtype=np.int64)


def apply_central_occlusion(images: np.ndarray, size: int = CENTRAL_OCCLUSION_SIZE) -> np.ndarray:
    """Aplica patch preto de `size`x`size` no centro de cada imagem."""
    out = images.copy()
    h, w = out.shape[1], out.shape[2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    out[:, y0:y0 + size, x0:x0 + size, :] = 0.0
    return out


def apply_distributed_occlusion(
    images: np.ndarray,
    patch: int = PATCH_SIZE,
    ratio: float = OCCLUSION_RATIO,
    seed: int = 42,
) -> np.ndarray:
    """Zera aleatoriamente `ratio` dos patches `patch`x`patch`.

    Cada imagem recebe sua propria mascara aleatoria, mas o seed garante
    reproducibilidade entre execucoes.
    """
    rng = np.random.default_rng(seed)
    out = images.copy()
    h, w = out.shape[1], out.shape[2]
    n_h = h // patch
    n_w = w // patch
    n_patches = n_h * n_w
    n_drop = int(round(ratio * n_patches))

    for i in range(out.shape[0]):
        idx = rng.choice(n_patches, size=n_drop, replace=False)
        for k in idx:
            r = (k // n_w) * patch
            c = (k % n_w) * patch
            out[i, r:r + patch, c:c + patch, :] = 0.0
    return out


def save_preview(
    images: np.ndarray,
    labels: np.ndarray,
    class_names,
    path: Path,
    title: str,
    n: int = 3,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img = np.clip(images[i].astype(np.float32) / 255.0, 0, 1)
        ax.imshow(img)
        name = class_names[int(labels[i])] if class_names is not None else int(labels[i])
        ax.set_title(f"Label: {name} ({int(labels[i])})", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def evaluate_accuracy(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    preprocess_fn=None,
    batch_size: int = 32,
) -> float:
    """Calcula acuracia top-1.

    `preprocess_fn` deve receber tensor float32 [0,255] e devolver o
    tensor pronto para o backbone (escala/normalizacao depende do modelo).
    """
    n = images.shape[0]
    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tf.constant(images[start:end], dtype=tf.float32)
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)
        logits = model(batch, training=False).numpy()
        preds = logits.argmax(axis=1)
        correct += int((preds == labels[start:end]).sum())
    return correct / n


def gpu_memory_growth() -> None:
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def silence_tf_logs() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
