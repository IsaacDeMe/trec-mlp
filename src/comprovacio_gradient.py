"""Comprovacio del gradient: derivada numerica contra la que calcula backward()."""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xarxa import Network

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "resultados")
EPS = 1e-6


def main():
    rng = np.random.default_rng(7)
    X = rng.random((4, 5))
    y = np.array([0, 1, 2, 1])

    net = Network([5, 4, 3], activation="sigmoid", init="xavier", seed=3, dtype=np.float64)

    p = net.forward(X)
    gW, gb = net.backward(X, y, p)

    def total_loss():
        return net.loss(net.forward(X), y)

    rows = []
    worst = 0.0

    for layer in range(len(net.W)):
        for idx in np.ndindex(net.W[layer].shape):
            orig = net.W[layer][idx]
            net.W[layer][idx] = orig + EPS
            lp = total_loss()
            net.W[layer][idx] = orig - EPS
            lm = total_loss()
            net.W[layer][idx] = orig
            num = (lp - lm) / (2 * EPS)
            ana = gW[layer][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
            rows.append((f"W{layer}{idx}", ana, num, rel))

        for idx in range(len(net.b[layer])):
            orig = net.b[layer][idx]
            net.b[layer][idx] = orig + EPS
            lp = total_loss()
            net.b[layer][idx] = orig - EPS
            lm = total_loss()
            net.b[layer][idx] = orig
            num = (lp - lm) / (2 * EPS)
            ana = gb[layer][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
            rows.append((f"b{layer}[{idx}]", ana, num, rel))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gradcheck.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "analitico", "numerico", "error_relativo"])
        for name, a, n, r in rows:
            w.writerow([name, f"{a:.10e}", f"{n:.10e}", f"{r:.3e}"])

    rels = np.array([r[3] for r in rows])
    print(f"parametres comprovats: {len(rows)}")
    print(f"error relatiu maxim:   {worst:.3e}")
    print(f"error relatiu mitja:   {rels.mean():.3e}")
    print("ok" if worst < 1e-6 else "revisa la retropropagacio")


if __name__ == "__main__":
    main()
