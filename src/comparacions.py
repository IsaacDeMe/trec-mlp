"""Els dos rivals de la xarxa: regressio logistica i k vei mes proper."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xarxa import Network


def knn_predict(Xtr, ytr, Xte, ks=(1, 3, 5), batch=500, verbose=False):
    """Calcula els veins per blocs i aprofita el mateix pas per a tots els k."""
    kmax = max(ks)
    preds = {k: np.empty(len(Xte), dtype=np.int64) for k in ks}
    tr2 = (Xtr * Xtr).sum(axis=1)
    t0 = time.perf_counter()

    for ini in range(0, len(Xte), batch):
        xb = Xte[ini:ini + batch]

        d = tr2[None, :] - 2.0 * (xb @ Xtr.T) + (xb * xb).sum(axis=1)[:, None]
        idx = np.argpartition(d, kmax, axis=1)[:, :kmax]
        labels = ytr[idx]
        for k in ks:
            votes = np.zeros((len(xb), 10), dtype=np.int32)
            for j in range(k):
                np.add.at(votes, (np.arange(len(xb)), labels[:, j]), 1)
            preds[k][ini:ini + batch] = votes.argmax(axis=1)

        if verbose and ini % (batch * 10) == 0:
            print(f"  {ini + len(xb)}/{len(Xte)}")

    return preds, time.perf_counter() - t0


def logistic(net=None, seed=0):
    if net is None:
        net = Network([784, 10], activation="sigmoid", init="xavier", seed=seed)
    return net


if __name__ == "__main__":
    from mnist import load_cached
    from xarxa import metrics

    d = load_cached()
    Xtr, ytr = d["X_train"], d["y_train"]


    t = time.perf_counter()
    preds, _ = knn_predict(Xtr, ytr, d["X_test"][:2000], ks=(1, 3, 5))
    dt = time.perf_counter() - t
    for k, p in preds.items():
        print(f"k={k}  acc={np.mean(p == d['y_test'][:2000]):.4f}")
    print(f"2.000 imatges en {dt:.1f} s  ->  10.000 trigarien unes {dt * 5:.0f} s")
