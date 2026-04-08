#!/usr/bin/env python3
"""
svm_sklearn_hog.py 2026

Pipeline:
1) inverter cores, bounding box e resize para 20x20
2) deskew
3) HOG manual
4) data augmentation opcional
5) SVM do scikit-learn
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


DEFAULT_MNIST_DIR = Path("/home/andre/cekeikon5/tiny_dnn/data")
DEFAULT_CSV_DIR = Path(__file__).resolve().parent / "assets" / "MNIST_CSV"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_VERSION = "restore_rbf_full_v1"


def read_idx_images(path: Path, limit: int | None = None) -> np.ndarray:
    with path.open("rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Magic invalido para imagens IDX: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
    return data[:limit] if limit is not None else data


def read_idx_labels(path: Path, limit: int | None = None) -> np.ndarray:
    with path.open("rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Magic invalido para rotulos IDX: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    data = data[:count]
    return data[:limit] if limit is not None else data


def read_csv_split(path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", dtype=np.uint8, max_rows=limit)
    y = data[:, 0].astype(np.int32)
    x = data[:, 1:].reshape(-1, 28, 28)
    return x, y


def load_mnist(mnist_dir: Path, train_limit: int | None, test_limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = mnist_dir / "train-images.idx3-ubyte"
    train_labels = mnist_dir / "train-labels.idx1-ubyte"
    test_images = mnist_dir / "t10k-images.idx3-ubyte"
    test_labels = mnist_dir / "t10k-labels.idx1-ubyte"

    if train_images.exists() and train_labels.exists() and test_images.exists() and test_labels.exists():
        ax = read_idx_images(train_images, train_limit)
        ay = read_idx_labels(train_labels, train_limit)
        qx = read_idx_images(test_images, test_limit)
        qy = read_idx_labels(test_labels, test_limit)
        return ax, ay, qx, qy

    csv_train = mnist_dir / "mnist_train.csv"
    csv_test = mnist_dir / "mnist_test.csv"
    if csv_train.exists() and csv_test.exists():
        ax, ay = read_csv_split(csv_train, train_limit)
        qx, qy = read_csv_split(csv_test, test_limit)
        return ax, ay, qx, qy

    raise FileNotFoundError(f"Nao encontrei MNIST em {mnist_dir}")


def resize_to_square(img: np.ndarray, side: int) -> np.ndarray:
    zoom_y = side / img.shape[0]
    zoom_x = side / img.shape[1]
    out = ndi.zoom(img, (zoom_y, zoom_x), order=1)
    out = out[:side, :side]
    if out.shape != (side, side):
        padded = np.full((side, side), out.max() if out.size else 1.0, dtype=np.float32)
        padded[: out.shape[0], : out.shape[1]] = out
        out = padded
    return out.astype(np.float32, copy=False)


def preprocess_base(img_u8: np.ndarray, side: int = 20) -> np.ndarray:
    img = 255.0 - img_u8.astype(np.float32)

    mask = img < 250.0
    if np.any(mask):
        ys, xs = np.where(mask)
        img = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    img = resize_to_square(img, side)
    return img / 255.0


def deskew(img: np.ndarray) -> np.ndarray:
    mass = 1.0 - img
    total = mass.sum()
    if total < 1e-6:
        return img

    y, x = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    cx = float((x * mass).sum() / total)
    cy = float((y * mass).sum() / total)
    x0 = x - cx
    y0 = y - cy
    mu11 = float((x0 * y0 * mass).sum())
    mu02 = float(((y0 ** 2) * mass).sum())
    if abs(mu02) < 1e-6:
        return img

    skew = mu11 / mu02
    transform = np.array([[1.0, 0.0], [skew, 1.0]], dtype=np.float32)
    offset = np.array([0.0, -0.5 * img.shape[0] * skew], dtype=np.float32)
    corrected = ndi.affine_transform(
        img,
        transform,
        offset=offset,
        order=1,
        mode="constant",
        cval=1.0,
    )
    return corrected.astype(np.float32, copy=False)


def shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.ones_like(img, dtype=np.float32)
    y0 = max(0, dy)
    y1 = img.shape[0] + min(0, dy)
    x0 = max(0, dx)
    x1 = img.shape[1] + min(0, dx)
    sy0 = max(0, -dy)
    sy1 = sy0 + (y1 - y0)
    sx0 = max(0, -dx)
    sx1 = sx0 + (x1 - x0)
    out[y0:y1, x0:x1] = img[sy0:sy1, sx0:sx1]
    return out


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    return ndi.rotate(img, angle=angle, reshape=False, order=1, mode="constant", cval=1.0).astype(np.float32)


def hog_descriptor(img: np.ndarray, cell: int = 4, bins: int = 9) -> np.ndarray:
    mass = 1.0 - img
    gy, gx = np.gradient(mass)
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0) / (180.0 / bins)

    cells_y = img.shape[0] // cell
    cells_x = img.shape[1] // cell
    hist = np.zeros((cells_y, cells_x, bins), dtype=np.float32)

    for cy in range(cells_y):
        for cx in range(cells_x):
            y0 = cy * cell
            y1 = y0 + cell
            x0 = cx * cell
            x1 = x0 + cell
            cell_mag = mag[y0:y1, x0:x1].ravel()
            cell_ang = ang[y0:y1, x0:x1].ravel()
            lower = np.floor(cell_ang).astype(np.int32) % bins
            upper = (lower + 1) % bins
            frac = cell_ang - np.floor(cell_ang)
            np.add.at(hist[cy, cx], lower, cell_mag * (1.0 - frac))
            np.add.at(hist[cy, cx], upper, cell_mag * frac)

    feats = []
    for cy in range(cells_y - 1):
        for cx in range(cells_x - 1):
            block = hist[cy : cy + 2, cx : cx + 2].ravel()
            norm = np.linalg.norm(block) + 1e-6
            block = block / norm
            block = np.clip(block, 0.0, 0.2)
            block = block / (np.linalg.norm(block) + 1e-6)
            feats.append(block)
    return np.concatenate(feats).astype(np.float32)


def feature_vector(img_u8: np.ndarray) -> np.ndarray:
    base = preprocess_base(img_u8, side=20)
    corrected = deskew(base)
    return hog_descriptor(corrected)


def augmented_feature_vectors(img_u8: np.ndarray, augment: str) -> list[np.ndarray]:
    base = preprocess_base(img_u8, side=20)
    variants = [base]

    if augment in {"shift", "full"}:
        variants.extend([
            shift(base, -1, 0),
            shift(base, 1, 0),
            shift(base, 0, -1),
            shift(base, 0, 1),
        ])
    if augment == "full":
        variants.extend([rotate(base, -7.0), rotate(base, 7.0)])

    return [hog_descriptor(deskew(v)) for v in variants]


def extract_train_features(images: np.ndarray, labels: np.ndarray, augment: str, jobs: int) -> tuple[np.ndarray, np.ndarray]:
    extracted = Parallel(n_jobs=jobs, prefer="processes")(
        delayed(augmented_feature_vectors)(img, augment) for img in images
    )
    feats = []
    ys = []
    for y, vecs in zip(labels, extracted):
        feats.extend(vecs)
        ys.extend([y] * len(vecs))
    return np.asarray(feats, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def extract_test_features(images: np.ndarray, jobs: int) -> np.ndarray:
    feats = Parallel(n_jobs=jobs, prefer="processes")(
        delayed(feature_vector)(img) for img in images
    )
    return np.asarray(feats, dtype=np.float32)


def cache_key(mnist_dir: Path, train_limit: int | None, test_limit: int | None, augment: str) -> str:
    text = f"{mnist_dir}|{train_limit}|{test_limit}|{augment}|side20|hog4x4|deskew_v2|{CACHE_VERSION}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def get_cached_features(cache_file: Path):
    if not cache_file.exists():
        return None
    data = np.load(cache_file)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def save_cached_features(cache_file: Path, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def build_model(model_name: str):
    if model_name == "linear":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=2.0, dual="auto", max_iter=30000),
        )
    if model_name == "rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(C=12.0, gamma="scale", kernel="rbf", cache_size=2048),
        )
    raise ValueError(f"Modelo desconhecido: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", type=Path, default=DEFAULT_MNIST_DIR)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--augment", choices=["none", "shift", "full"], default="full")
    parser.add_argument("--model", choices=["linear", "rbf"], default="rbf")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    mnist_dir = args.mnist_dir
    if not mnist_dir.exists() and DEFAULT_CSV_DIR.exists():
        mnist_dir = DEFAULT_CSV_DIR

    t0 = time.time()
    cache_file = RESULTS_DIR / f"mnist_feat_{cache_key(mnist_dir, args.train_limit, args.test_limit, args.augment)}.npz"
    cached = None if args.no_cache else get_cached_features(cache_file)

    if cached is None:
        ax, ay, qx, qy = load_mnist(mnist_dir, args.train_limit, args.test_limit)
        x_train, y_train = extract_train_features(ax, ay, args.augment, args.jobs)
        x_test = extract_test_features(qx, args.jobs)
        y_test = qy.astype(np.int32)
        if not args.no_cache:
            save_cached_features(cache_file, x_train, y_train, x_test, y_test)
        cache_status = "novo"
    else:
        x_train, y_train, x_test, y_test = cached
        cache_status = "existente"

    t1 = time.time()
    clf = build_model(args.model)
    clf.fit(x_train, y_train)
    t2 = time.time()
    pred = clf.predict(x_test)
    t3 = time.time()

    err = 100.0 * np.count_nonzero(pred != y_test) / y_test.shape[0]
    print(f"Erros={err:10.2f}%")
    print(f"Tempo de preprocessamento: {t1 - t0:.6f}")
    print(f"Tempo de treino: {t2 - t1:.6f}")
    print(f"Tempo de predicao: {t3 - t2:.6f}")


if __name__ == "__main__":
    main()
