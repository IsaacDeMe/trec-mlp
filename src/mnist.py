"""Baixada i carrega de MNIST directament dels fitxers idx, sense llibreries de IA."""

import gzip
import os
import urllib.request

import numpy as np

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
NPZ = os.path.join(DATA_DIR, "mnist.npz")

VAL_SIZE = 6000
SEED = 42


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            continue
        print(f"descarregant {name} ...")
        urllib.request.urlretrieve(BASE + name, path)


def _read_images(path):
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, rows * cols)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        _, n = np.frombuffer(f.read(8), dtype=">i4")
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)


def load():
    """Retorna train, val i test ja normalitzats entre 0 i 1."""
    download()

    X = _read_images(os.path.join(DATA_DIR, FILES[0])).astype(np.float32) / 255.0
    y = _read_labels(os.path.join(DATA_DIR, FILES[1]))
    Xte = _read_images(os.path.join(DATA_DIR, FILES[2])).astype(np.float32) / 255.0
    yte = _read_labels(os.path.join(DATA_DIR, FILES[3]))


    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    val_idx = perm[:VAL_SIZE]
    train_idx = perm[VAL_SIZE:]

    data = {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": Xte, "y_test": yte,
    }
    np.savez_compressed(NPZ, **data)
    return data


def load_cached():
    if not os.path.exists(NPZ):
        return load()
    with np.load(NPZ) as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    d = load()
    for k, v in d.items():
        print(k, v.shape, v.dtype)
    print("pixels min/max:", d["X_train"].min(), d["X_train"].max())
    print("etiquetes val:", np.bincount(d["y_val"]))
