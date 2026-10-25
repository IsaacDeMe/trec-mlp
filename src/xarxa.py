"""La xarxa, escrita a ma: forward, retropropagacio i descens del gradient."""

import time

import numpy as np


def sigmoid(z):

    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    with np.errstate(over="ignore", invalid="ignore"):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class Network:
    def __init__(self, layers, activation="relu", init="he", seed=0, dtype=np.float32):
        self.layers = list(layers)
        self.activation = activation
        self.init = init
        rng = np.random.default_rng(seed)

        self.W, self.b = [], []
        for fan_in, fan_out in zip(layers[:-1], layers[1:]):
            if init == "zeros":
                w = np.zeros((fan_in, fan_out))
            elif init == "he":
                w = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            elif init == "xavier":
                w = rng.normal(0.0, np.sqrt(1.0 / fan_in), (fan_in, fan_out))
            else:
                w = rng.normal(0.0, 0.01, (fan_in, fan_out))
            self.W.append(w.astype(dtype))
            self.b.append(np.zeros(fan_out, dtype=dtype))

        self._z = []
        self._a = []

    def forward(self, X):
        a = X
        self._z, self._a = [], [X]
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            self._z.append(z)
            if i == last:
                a = softmax(z)
            elif self.activation == "relu":
                a = relu(z)
            else:
                a = sigmoid(z)
            self._a.append(a)
        return a

    def loss(self, p, y):
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

    def backward(self, X, y, p):
        m = len(X)
        d = p.copy()
        d[np.arange(m), y] -= 1.0
        d /= m

        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = self._a[i].T @ d
            gb[i] = d.sum(axis=0)
            if i == 0:
                continue
            d = d @ self.W[i].T
            z = self._z[i - 1]
            if self.activation == "relu":
                d = d * (z > 0)
            else:
                s = sigmoid(z)
                d = d * s * (1.0 - s)
        return gW, gb

    def predict_proba(self, X, batch=2048):
        out = np.empty((len(X), self.layers[-1]), dtype=np.float32)
        for ini in range(0, len(X), batch):
            out[ini:ini + batch] = self.forward(X[ini:ini + batch])
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    @property
    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    def fit(self, X, y, X_val=None, y_val=None, epochs=10, batch=100, lr=0.1,
            momentum=0.9, seed=0, eval_each=1, keep_best=True, verbose=False):
        rng = np.random.default_rng(seed)
        vW = [np.zeros_like(w) for w in self.W]
        vb = [np.zeros_like(b) for b in self.b]

        hist = {"epoch": [], "loss_train": [], "acc_train": [],
                "loss_val": [], "acc_val": []}
        best_acc, best_W, best_b = -1.0, None, None
        t0 = time.perf_counter()

        for ep in range(1, epochs + 1):
            order = rng.permutation(len(X))
            for ini in range(0, len(X), batch):
                idx = order[ini:ini + batch]
                p = self.forward(X[idx])
                gW, gb = self.backward(X[idx], y[idx], p)
                for i in range(len(self.W)):
                    vW[i] = momentum * vW[i] + gW[i]
                    vb[i] = momentum * vb[i] + gb[i]
                    self.W[i] -= lr * vW[i]
                    self.b[i] -= lr * vb[i]

            if ep % eval_each == 0 or ep == epochs:
                p = self.forward(X)
                acc = float((p.argmax(1) == y).mean())
                row = {"epoch": ep, "loss_train": self.loss(p, y), "acc_train": acc}

                if X_val is not None:
                    pv = self.predict_proba(X_val)
                    acc_v = float((pv.argmax(1) == y_val).mean())
                    row["loss_val"] = self.loss(pv, y_val)
                    row["acc_val"] = acc_v
                    if acc_v > best_acc:
                        best_acc = acc_v
                        best_W = [w.copy() for w in self.W]
                        best_b = [b.copy() for b in self.b]
                else:
                    row["loss_val"] = float("nan")
                    row["acc_val"] = float("nan")

                for k, v in row.items():
                    hist[k].append(v)

            if verbose and (ep % 5 == 0 or ep == epochs):
                print(f"  epoca {ep:3d}  loss {hist['loss_train'][-1]:.4f}  "
                      f"acc {hist['acc_train'][-1]:.4f}  val {hist['acc_val'][-1]:.4f}")

        if keep_best and best_W is not None:
            self.W, self.b = best_W, best_b

        hist["time"] = time.perf_counter() - t0
        return hist

    def save(self, path):
        arrays = {"capas": np.array(self.layers), "activacion": np.array(self.activation)}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            layers = f["capas"].tolist()
            net = cls(layers, activation=str(f["activacion"]))
            net.W = [f[f"W{i}"] for i in range(len(layers) - 1)]
            net.b = [f[f"b{i}"] for i in range(len(layers) - 1)]
        return net


def confusion(y, pred, n=10):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def metrics(y, pred):
    n = int(max(y.max(), pred.max())) + 1
    cm = confusion(y, pred, n)
    acc = np.trace(cm) / cm.sum()
    precision = np.diag(cm) / np.clip(cm.sum(0), 1, None)
    recall = np.diag(cm) / np.clip(cm.sum(1), 1, None)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return {
        "acc": float(acc),
        "f1_macro": float(f1.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
    }
