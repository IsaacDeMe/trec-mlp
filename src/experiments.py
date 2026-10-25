"""Tots els experiments del treball, en ordre. Escriu els resultats a resultados/."""

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comparacions
from mnist import load_cached
from xarxa import Network, metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "resultats")
MODELS = os.path.join(ROOT, "models")
SEED = 0
LAYERS = [784, 16, 16, 10]


def save_csv(name, rows, fields=None):
    os.makedirs(OUT, exist_ok=True)
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def search_lr(d):
    """Un learning rate per activacio, triat amb validacio (no amb test)."""
    rows, best = [], {}
    for act, init in (("sigmoid", "xavier"), ("relu", "he")):
        for lr in (0.1, 0.5, 1.0):
            net = Network(LAYERS, act, init, seed=SEED)
            h = net.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                        epochs=4, lr=lr, seed=SEED, eval_each=4)
            acc = float(np.nanmax(h["acc_val"]))
            rows.append({"activacion": act, "lr": lr, "acc_val": round(acc, 4)})
            print(f"  {act:8s} lr={lr:<4} val={acc:.4f}")
            if act not in best or acc > best[act][0]:
                best[act] = (acc, lr)
    save_csv("lr_search.csv", rows)
    return {a: v[1] for a, v in best.items()}, {a: v[0] for a, v in best.items()}


def curves(d, lr):
    """Les dues corbes d'aprenentatge i el model principal."""
    rows, summary = [], []
    for act, init in (("sigmoid", "xavier"), ("relu", "he")):
        net = Network(LAYERS, act, init, seed=SEED)
        h = net.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                    epochs=20, lr=lr[act], seed=SEED, verbose=True)
        for i in range(len(h["epoch"])):
            rows.append({
                "modelo": act, "epoca": h["epoch"][i],
                "loss_train": round(h["loss_train"][i], 5),
                "acc_train": round(h["acc_train"][i], 5),
                "loss_val": round(h["loss_val"][i], 5),
                "acc_val": round(h["acc_val"][i], 5),
            })

        p = net.predict(d["X_test"])
        m = metrics(d["y_test"], p)
        summary.append({
            "modelo": f"mlp_16_16_{act}", "acc_val": round(float(np.nanmax(h["acc_val"])), 4),
            "acc_test": round(m["acc"], 4), "f1_macro": round(m["f1_macro"], 4),
            "params": net.n_params, "tiempo_train": round(h["time"], 2),
        })
        net.save(os.path.join(MODELS, f"mlp_{act}.npz"))
        print(f"  {act}: test {m['acc']:.4f}  f1 {m['f1_macro']:.4f}  "
              f"({h['time']:.1f}s)  params {net.n_params}")

    save_csv("curvas.csv", rows)
    return summary


def ablation(d, lr):
    rows = []

    def run_experiment(name, config, layers=LAYERS, act="relu", init="he", lr_uso=None, ep=10):
        net = Network(layers, act, init, seed=SEED)
        h = net.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                    epochs=ep, lr=lr_uso or lr[act], seed=SEED, eval_each=ep)
        p = net.predict(d["X_test"])
        m = metrics(d["y_test"], p)
        acc_v = float(np.nanmax(h["acc_val"]))
        rows.append({"experimento": name, "configuracion": config,
                     "acc_val": round(acc_v, 4), "acc_test": round(m["acc"], 4),
                     "params": net.n_params, "tiempo": round(h["time"], 2)})
        print(f"  {name:14s} {config:22s} val={acc_v:.4f} test={m['acc']:.4f}")


    for act, init in (("sigmoid", "xavier"), ("relu", "he")):
        run_experiment("activacion", act, act=act, init=init)

    for init in ("zeros", "normal", "he"):
        run_experiment("inicializacion", init, init=init)

    for lr_uso in (0.01, 0.1, 0.5, 1.0):
        run_experiment("learning_rate", str(lr_uso), lr_uso=lr_uso)

    save_csv("ablacion.csv", rows)
    return rows


def data_efficiency(d):
    """Mateix pressupost d'actualitzacions; l'unica cosa que canvia son les dades."""
    rows = []
    sizes = [100, 500, 2000, 10000, 30000, 54000]
    steps = 10800
    batch = 100

    for n in sizes:
        for seed in (0, 1, 2):
            if n == len(d["X_train"]):
                X, y = d["X_train"], d["y_train"]
            else:
                rng = np.random.default_rng(100 + seed)
                idx = rng.choice(len(d["X_train"]), n, replace=False)
                X, y = d["X_train"][idx], d["y_train"][idx]

            epochs = max(1, round(steps / max(1, len(X) // batch)))
            net = Network(LAYERS, "relu", "he", seed=seed)
            h = net.fit(X, y, d["X_val"], d["y_val"], epochs=epochs, lr=0.1,
                        seed=seed, eval_each=max(1, epochs // 20))
            acc = float((net.predict(d["X_test"]) == d["y_test"]).mean())
            rows.append({"n": n, "seed": seed, "epocas": epochs, "acc_test": round(acc, 4)})
        accs = [r["acc_test"] for r in rows if r["n"] == n]
        print(f"  n={n:6d}  acc={np.mean(accs):.4f} +- {np.std(accs):.4f}")

    save_csv("datos.csv", rows)
    return rows


def shift(X, dx, dy):
    im = X.reshape(-1, 28, 28)
    out = np.zeros_like(im)
    y0, y1 = max(0, dy), min(28, 28 + dy)
    x0, x1 = max(0, dx), min(28, 28 + dx)
    sy0, sy1 = max(0, -dy), min(28, 28 - dy)
    sx0, sx1 = max(0, -dx), min(28, 28 - dx)
    out[:, y0:y1, x0:x1] = im[:, sy0:sy1, sx0:sx1]
    return out.reshape(len(X), -1)


def rotate(X, angle):
    from PIL import Image

    im = (X.reshape(-1, 28, 28) * 255).astype(np.uint8)
    out = np.empty_like(im)
    for i in range(len(im)):
        out[i] = np.array(Image.fromarray(im[i]).rotate(
            angle, resample=Image.BILINEAR, fillcolor=0))
    return out.reshape(len(X), -1).astype(np.float32) / 255.0


def evaluation(d, net, curve_rows):
    Xte, yte = d["X_test"], d["y_test"]

    t0 = time.perf_counter()
    proba = net.predict_proba(Xte)
    t_inf = time.perf_counter() - t0

    pred = proba.argmax(axis=1)
    conf = proba[np.arange(len(pred)), pred]
    m = metrics(yte, pred)

    print(f"  test acc={m['acc']:.4f}  f1={m['f1_macro']:.4f}  "
          f"inferencia 10k={t_inf:.2f}s")

    with open(os.path.join(OUT, "confusion.csv"), "w", encoding="utf-8") as f:
        f.write("real,predicho,n\n")
        for i in range(10):
            for j in range(10):
                f.write(f"{i},{j},{m['cm'][i, j]}\n")

    with open(os.path.join(OUT, "por_clase.csv"), "w", encoding="utf-8") as f:
        f.write("digito,precision,recall,f1,soporte\n")
        for i in range(10):
            f.write(f"{i},{m['precision'][i]:.4f},{m['recall'][i]:.4f},"
                    f"{m['f1'][i]:.4f},{int(m['cm'][i].sum())}\n")

    wrong = np.where(pred != yte)[0]
    orden = wrong[np.argsort(-conf[wrong])][:20]
    with open(os.path.join(OUT, "errors.csv"), "w", encoding="utf-8") as f:
        f.write("indice,real,predicho,confianza\n")
        for i in orden:
            f.write(f"{i},{yte[i]},{pred[i]},{conf[i]:.4f}\n")


    rng = np.random.default_rng(123)
    noise = rng.random((5, 784)).astype(np.float32)
    pn = net.predict_proba(noise)
    with open(os.path.join(OUT, "soroll.csv"), "w", encoding="utf-8") as f:
        f.write("imagen,prediccion,confianza\n")
        for i in range(5):
            f.write(f"{i},{pn[i].argmax()},{pn[i].max():.4f}\n")
    np.savez(os.path.join(OUT, "soroll.npz"), images=noise, proba=pn)


    rows = []
    for sigma in (0.0, 0.1, 0.2, 0.3, 0.5):
        r = np.random.default_rng(7)
        Xn = np.clip(Xte + r.normal(0, sigma, Xte.shape).astype(np.float32), 0, 1)
        acc = float((net.predict(Xn) == yte).mean())
        rows.append({"tipo": "soroll", "parametro": sigma, "acc": round(acc, 4)})
        print(f"  soroll sigma={sigma}: {acc:.4f}")

    for dx in (-4, -2, -1, 0, 1, 2, 4):
        acc = float((net.predict(shift(Xte, dx, 0)) == yte).mean())
        rows.append({"tipo": "desplazamiento_x", "parametro": dx, "acc": round(acc, 4)})
        print(f"  desplazament {dx:+d}px: {acc:.4f}")

    sub = np.random.default_rng(11).choice(len(Xte), 3000, replace=False)
    Xs, ys = Xte[sub], yte[sub]
    for ang in (-20, -10, -5, 0, 5, 10, 20):
        acc = float((net.predict(rotate(Xs, ang)) == ys).mean())
        rows.append({"tipo": "rotacion", "parametro": ang, "acc": round(acc, 4)})
        print(f"  rotacio {ang:+d} graus: {acc:.4f}")

    save_csv("robustez.csv", rows)


    np.savez(os.path.join(OUT, "pesos.npz"), W1=net.W[0])

    summary = {
        "acc_test": round(m["acc"], 4),
        "f1_macro": round(m["f1_macro"], 4),
        "params": net.n_params,
        "tiempo_inferencia_10k": round(t_inf, 3),
        "precision": [round(float(v), 4) for v in m["precision"]],
        "recall": [round(float(v), 4) for v in m["recall"]],
        "f1": [round(float(v), 4) for v in m["f1"]],
        "matriz": m["cm"].tolist(),
    }
    with open(os.path.join(OUT, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, t_inf


def comparison(d, resumen_mlp, curves_summary, lr):
    rows = []


    best = (-1, 0.1)
    for lr_log in (0.1, 0.5, 1.0):
        net = Network([784, 10], "sigmoid", "xavier", seed=SEED)
        h = net.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                    epochs=8, lr=lr_log, seed=SEED, eval_each=8)
        acc = float(np.nanmax(h["acc_val"]))
        if acc > best[0]:
            best = (acc, lr_log)
    print(f"  logistica lr={best[1]} (val {best[0]:.4f})")

    net = Network([784, 10], "sigmoid", "xavier", seed=SEED)
    t0 = time.perf_counter()
    h = net.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                epochs=20, lr=best[1], seed=SEED)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = net.predict(d["X_test"])
    t_inf = time.perf_counter() - t0
    m = metrics(d["y_test"], p)
    rows.append({"modelo": "logistica", "acc_test": round(m["acc"], 4),
                 "f1_macro": round(m["f1_macro"], 4), "params": net.n_params,
                 "memoria_mb": round(net.n_params * 4 / 1e6, 2),
                 "tiempo_train": round(t_train, 2), "tiempo_inferencia": round(t_inf, 3)})
    net.save(os.path.join(MODELS, "logistica.npz"))
    print(f"  logistica: test {m['acc']:.4f} ({t_inf:.2f}s inferencia)")


    Xtr, ytr = d["X_train"], d["y_train"]
    preds_val, t_val = comparacions.knn_predict(Xtr, ytr, d["X_val"], ks=(1, 3, 5))
    accs_val = {k: float((preds_val[k] == d["y_val"]).mean()) for k in preds_val}
    best_k = max(accs_val, key=accs_val.get)
    print(f"  kNN val: {accs_val} -> k={best_k}")

    preds, t_test = comparacions.knn_predict(Xtr, ytr, d["X_test"], ks=(1, 3, 5))
    for k in preds:
        m = metrics(d["y_test"], preds[k])

        rows.append({"modelo": f"knn_k{k}", "acc_test": round(m["acc"], 4),
                     "f1_macro": round(m["f1_macro"], 4), "params": 0,
                     "memoria_mb": round(Xtr.nbytes / 1e6, 2),
                     "tiempo_train": 0.0,
                     "tiempo_inferencia": round(t_test, 3)})
        print(f"  kNN k={k}: test {m['acc']:.4f}")


    fila = next(r for r in curves_summary if r["modelo"] == "mlp_16_16_relu")
    rows.append({"modelo": "mlp_16_16", "acc_test": resumen_mlp["acc_test"],
                 "f1_macro": resumen_mlp["f1_macro"], "params": resumen_mlp["params"],
                 "memoria_mb": round(resumen_mlp["params"] * 4 / 1e6, 3),
                 "tiempo_train": fila["tiempo_train"],
                 "tiempo_inferencia": resumen_mlp["tiempo_inferencia_10k"]})

    for name, layers in (("mlp_64", [784, 64, 10]),
                          ("mlp_128_64", [784, 128, 64, 10]),
                          ("mlp_256_128", [784, 256, 128, 10])):
        net = Network(layers, "relu", "he", seed=SEED)
        t0 = time.perf_counter()
        net.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                epochs=20, lr=lr["relu"], seed=SEED)
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = net.predict(d["X_test"])
        t_inf = time.perf_counter() - t0
        m = metrics(d["y_test"], p)
        rows.append({"modelo": name, "acc_test": round(m["acc"], 4),
                     "f1_macro": round(m["f1_macro"], 4), "params": net.n_params,
                     "memoria_mb": round(net.n_params * 4 / 1e6, 2),
                     "tiempo_train": round(t_train, 2),
                     "tiempo_inferencia": round(t_inf, 3)})
        net.save(os.path.join(MODELS, f"{name}.npz"))
        print(f"  {name}: test {m['acc']:.4f}  params {net.n_params}  ({t_train:.1f}s)")

    save_csv("comparacion.csv", rows)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    d = load_cached()
    print("dades carregades")

    print("\n[1] learning rate per activacio")
    lr, acc_lr = search_lr(d)

    print("\n[2] corbes d'aprenentatge")
    summary = curves(d, lr)

    print("\n[3] ablacio")
    ablation(d, lr)

    print("\n[4] eficiencia de dades")
    data_efficiency(d)

    print("\n[5] avaluacio del model principal")
    net = Network.load(os.path.join(MODELS, "mlp_relu.npz"))
    resumen_mlp, t_inf = evaluation(d, net, read_csv("curvas.csv"))

    print("\n[6] comparacio amb logistica i kNN")
    comparison(d, resumen_mlp, summary, lr)

    print("\nllestos. Els resultats son a resultados/")


if __name__ == "__main__":
    main()
