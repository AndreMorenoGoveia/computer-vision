"""Avalia os tres modelos treinados sob as 3 condicoes de teste.

Para cada modelo, calcula acuracia em:
  - Teste original (sem oclusao)
  - Oclusao central (patch preto 112x112)
  - Oclusao distribuida (25% dos patches 16x16 zerados aleatoriamente)

Tambem salva as previews exigidas pelo enunciado:
  - Treino: 3 primeiras imagens
  - Teste: 3 primeiras imagens (limpas, com oclusao central, com distribuida)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import (
    MODELS_DIR,
    RESULTS_DIR,
    apply_central_occlusion,
    apply_distributed_occlusion,
    evaluate_accuracy,
    gpu_memory_growth,
    load_oxford_flowers,
    save_preview,
    silence_tf_logs,
)

silence_tf_logs()
gpu_memory_growth()


def evaluate_model(name: str, model_path: Path, datasets) -> dict:
    test_x, test_y = datasets["clean"]
    test_central = datasets["central"]
    test_distributed = datasets["distributed"]

    if not model_path.exists():
        print(f"[skip] {name}: modelo nao encontrado em {model_path}")
        return {"model": name, "missing": True}

    model = tf.keras.models.load_model(model_path)

    acc_clean = evaluate_accuracy(model, test_x, test_y)
    acc_central = evaluate_accuracy(model, test_central, test_y)
    acc_dist = evaluate_accuracy(model, test_distributed, test_y)

    print(f"{name}: clean={acc_clean:.4f}  central={acc_central:.4f}  distributed={acc_dist:.4f}")
    return {
        "model": name,
        "test_accuracy_clean": acc_clean,
        "test_accuracy_central": acc_central,
        "test_accuracy_distributed": acc_dist,
    }


def main() -> None:
    train_ds, _, (test_x, test_y), info = load_oxford_flowers(batch_size=32, augment=False)
    class_names = info.features["label"].names if info is not None else None

    test_central = apply_central_occlusion(test_x)
    test_distributed = apply_distributed_occlusion(test_x)

    save_preview(
        images=test_x[:3],
        labels=test_y[:3],
        class_names=class_names,
        path=RESULTS_DIR / "test_clean_preview.png",
        title="Teste - Primeiras 3 imagens sem oclusao",
    )
    save_preview(
        images=test_central[:3],
        labels=test_y[:3],
        class_names=class_names,
        path=RESULTS_DIR / "test_central_preview.png",
        title="Teste - Primeiras 3 imagens com oclusao central",
    )
    save_preview(
        images=test_distributed[:3],
        labels=test_y[:3],
        class_names=class_names,
        path=RESULTS_DIR / "test_distributed_preview.png",
        title="Teste - Primeiras 3 imagens com oclusao distribuida",
    )

    train_imgs, train_labels = [], []
    for batch_x, batch_y in train_ds.take(1):
        train_imgs.append(batch_x.numpy()[:3])
        train_labels.append(batch_y.numpy()[:3])
    save_preview(
        images=train_imgs[0],
        labels=train_labels[0],
        class_names=class_names,
        path=RESULTS_DIR / "train_preview.png",
        title="Treino - Primeiras 3 imagens",
    )

    datasets = {
        "clean": (test_x, test_y),
        "central": test_central,
        "distributed": test_distributed,
    }

    rows = []
    for name, fname in [
        ("EfficientNetB0", "cnn_best.keras"),
        ("DeiT-Tiny", "vit_best.keras"),
        ("Best (ViT-Base)", "best_model.keras"),
    ]:
        rows.append(evaluate_model(name, MODELS_DIR / fname, datasets))

    out = RESULTS_DIR / "final_metrics.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Resultados salvos em {out}")


if __name__ == "__main__":
    main()
