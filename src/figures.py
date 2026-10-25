"""Figures del treball a partir dels CSV de resultados/."""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnist import load_cached

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "resultados")
FIG = os.path.join(ROOT, "figuras")
MOD = os.path.join(ROOT, "modelos")

AZUL = "#2b6cb0"
NARANJA = "#c05621"
VERDE = "#2f855a"
MORADO = "#6b46c1"
GRIS = "#718096"

os.makedirs(FIG, exist_ok=True)


def style():
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def read_csv_rows(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_figure(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figura", name)


def fig_curves():
    rows = read_csv_rows("curvas.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    for act, color in (("sigmoid", NARANJA), ("relu", AZUL)):
        r = [x for x in rows if x["modelo"] == act]
        ep = [int(x["epoca"]) for x in r]
        tr = [float(x["loss_train"]) for x in r]
        va = [float(x["loss_val"]) for x in r]
        axes[0].plot(ep, tr, color=color, lw=1.6, label=f"{act} · entrenament")
        axes[0].plot(ep, va, color=color, lw=1.4, ls="--", marker="o", ms=2.5,
                     markevery=3, label=f"{act} · validació")
        axes[1].plot(ep, [float(x["acc_train"]) for x in r], color=color, lw=1.6)
        axes[1].plot(ep, [float(x["acc_val"]) for x in r], color=color, lw=1.4,
                     ls="--", marker="o", ms=2.5, markevery=3)

    axes[0].set_title("pèrdua (entropia creuada)")
    axes[0].set_xlabel("època")
    axes[0].set_ylabel("pèrdua")
    axes[1].set_title("exactitud")
    axes[1].set_xlabel("època")
    axes[1].set_ylabel("exactitud")
    axes[1].set_ylim(0.90, 0.97)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save_figure(fig, "f1_corbes.png")


def fig_confusion():
    rows = read_csv_rows("confusion.csv")
    cm = np.zeros((10, 10), dtype=int)
    for r in rows:
        cm[int(r["real"]), int(r["predicho"])] = int(r["n"])

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicció")
    ax.set_ylabel("dígit real")
    ax.grid(False)

    maxv = cm.max()
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v == 0:
                continue
            color = "white" if v > maxv * 0.55 else "#1a202c"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5, color=color)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("imatges", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(fig, "f2_confusio.png")


def fig_errors():
    rows = read_csv_rows("errors.csv")[:10]
    d = load_cached()
    X = d["X_test"]

    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.1))
    for ax, r in zip(axes.ravel(), rows):
        i = int(r["indice"])
        ax.imshow(X[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{r['real']} → {r['predicho']}  ({float(r['confianza']) * 100:.1f}%)",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "f3_errors.png")


def fig_ablation():
    rows = read_csv_rows("ablacion.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))

    def bars(ax, titulo, data, colores=None):
        labels = [d[0] for d in data]
        valores = [d[1] * 100 for d in data]
        colores = colores or [AZUL] * len(data)
        b = ax.bar(range(len(data)), valores, color=colores, width=0.62)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels)
        ax.axhline(10, color=GRIS, ls=":", lw=1)
        ax.set_title(titulo)
        ax.set_ylim(0, 100)
        for rect, v in zip(b, valores):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.0f}",
                    ha="center", fontsize=7)

    act = [(r["configuracion"], float(r["acc_test"])) for r in rows
           if r["experimento"] == "activacion"]
    ini = [(r["configuracion"], float(r["acc_test"])) for r in rows
           if r["experimento"] == "inicializacion"]
    lr = [(r["configuracion"], float(r["acc_test"])) for r in rows
          if r["experimento"] == "learning_rate"]

    bars(axes[0], "funció d'activació", act, [NARANJA, AZUL])
    bars(axes[1], "inicialització dels pesos", ini)
    bars(axes[2], "learning rate", lr)
    axes[0].set_ylabel("exactitud al test (%)")
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    fig.tight_layout()
    save_figure(fig, "f4_ablacio.png")


def fig_data():
    rows = read_csv_rows("datos.csv")
    ns = sorted({int(r["n"]) for r in rows})
    medias = [np.mean([float(r["acc_test"]) for r in rows if int(r["n"]) == n]) for n in ns]
    stds = [np.std([float(r["acc_test"]) for r in rows if int(r["n"]) == n]) for n in ns]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.errorbar(ns, medias, yerr=stds, color=AZUL, marker="o", ms=4.5,
                lw=1.6, capsize=3, label="xarxa 16·16 (mitjana de 3 llavors)")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(["100", "500", "2k", "10k", "30k", "54k"])
    ax.set_xlabel("imatges d'entrenament")
    ax.set_ylabel("exactitud al test")
    ax.set_ylim(0.63, 1.01)
    ax.set_xlim(ns[0] * 0.75, ns[-1] * 1.6)

    for i, (x, y) in enumerate(zip(ns, medias)):
        if i == len(ns) - 1:
            ax.annotate(f"{y * 100:.1f}%", (x, y), textcoords="offset points",
                        xytext=(6, 0), ha="left", va="center", fontsize=7)
        else:
            ax.annotate(f"{y * 100:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7)

    ax.axhline(0.9182, color=NARANJA, ls="--", lw=1.1)
    ax.text(ns[0] * 1.1, 0.925, "regressió logística (91,8%)", fontsize=7,
            color=NARANJA, va="bottom")
    ax.axhline(0.97, color=VERDE, ls=":", lw=1.2)
    ax.text(ns[0] * 1.1, 0.974, "kNN (97,0%)", fontsize=7, color=VERDE, va="bottom")

    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    save_figure(fig, "f5_dades.png")


def fig_robustness():
    rows = read_csv_rows("robustez.csv")
    tipos = [
        ("soroll", "soroll gaussià", "desviació $\\sigma$"),
        ("desplazamiento_x", "desplaçament horitzontal", "píxels"),
        ("rotacion", "rotació", "graus"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)

    for ax, (key, titulo, xlabel) in zip(axes, tipos):
        r = [x for x in rows if x["tipo"] == key]
        x = [float(v["parametro"]) for v in r]
        y = [float(v["acc"]) for v in r]
        ax.plot(x, y, color=AZUL, marker="o", ms=3.5, lw=1.5)
        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("exactitud")
    axes[0].set_xticks([0, 0.1, 0.2, 0.3, 0.5])
    axes[1].set_xticks([-4, -2, 0, 2, 4])
    axes[2].set_xticks([-20, -10, 0, 10, 20])
    axes[2].text(0.03, 0.06, "3.000 imatges", transform=axes[2].transAxes, fontsize=7,
                 color=GRIS)
    fig.tight_layout()
    save_figure(fig, "f6_robustesa.png")


def fig_noise():
    z = np.load(os.path.join(RES, "soroll.npz"))
    imgs, proba = z["imagenes"], z["proba"]
    table_rows = read_csv_rows("soroll.csv")

    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.7))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"→ {table_rows[i]['prediccion']}\n{float(table_rows[i]['confianza']) * 100:.1f}%",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "f7_soroll.png")


def fig_weights():
    z = np.load(os.path.join(RES, "pesos.npz"))
    W1 = z["W1"]
    n = W1.shape[1]
    vmax = np.percentile(np.abs(W1), 99.5)

    fig, axes = plt.subplots(4, 4, figsize=(4.6, 4.9))
    im = None
    for j, ax in enumerate(axes.ravel()):
        if j >= n:
            ax.axis("off")
            continue
        im = ax.imshow(W1[:, j].reshape(28, 28), cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{j + 1}", fontsize=6.5, pad=1)

    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("pes", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save_figure(fig, "f8_pesos.png")


def fig_weights2():
    z = np.load(os.path.join(MOD, "mlp_relu.npz"))
    W0, W1 = z["W0"], z["W1"]
    proj = W0 @ W1
    vmax = np.percentile(np.abs(proj), 99)

    fig, axes = plt.subplots(4, 4, figsize=(4.6, 4.9))
    im = None
    for j, ax in enumerate(axes.ravel()):
        im = ax.imshow(proj[:, j].reshape(28, 28), cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{j + 1}", fontsize=6.5, pad=1)

    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("pes projectat", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save_figure(fig, "f8b_pesos2.png")


def digit_grid(raw_images, clean_images, pred_raw, conf_raw, pred_net, conf_net,
                    digits, name):
    n = len(digits)
    fig, axes = plt.subplots(2, n, figsize=(0.7 * n + 0.5, 2.1))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j in range(n):
        for i, (img, pred, conf) in enumerate([
                (raw_images[j], pred_raw[j], conf_raw[j]),
                (clean_images[j], pred_net[j], conf_net[j])]):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            color = "black" if pred == digits[j] else "#c53030"
            ax.set_title(f"→ {pred}  {conf * 100:.0f}%", fontsize=7, color=color, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        axes[1, j].set_xlabel(str(digits[j]), fontsize=8, labelpad=2)

    axes[0, 0].set_ylabel("tal qual", fontsize=8)
    axes[1, 0].set_ylabel("preprocessat", fontsize=8)
    fig.tight_layout()
    save_figure(fig, name)


def fig_preprocess():
    z = np.load(os.path.join(RES, "preprocess_test.npz"))
    pc, pn = z["proba_cru"], z["proba_net"]
    digit_grid(z["crudos"], z["limpios"], pc.argmax(1), pc.max(1),
                    pn.argmax(1), pn.max(1), list(range(10)), "f9_preprocess.png")


def fig_handwriting():
    z = np.load(os.path.join(RES, "lletra.npz"))
    table_rows = {int(r["digito"]): r for r in read_csv_rows("lletra.csv")}
    digits = [int(d) for d in z["digitos"]]
    pc = np.array([int(table_rows[d]["cru_256"]) for d in digits])
    cc = np.array([float(table_rows[d]["conf_cru_256"]) for d in digits])
    pn = np.array([int(table_rows[d]["net_256"]) for d in digits])
    cn = np.array([float(table_rows[d]["conf_net_256"]) for d in digits])
    digit_grid(z["crudos"], z["limpios"], pc, cc, pn, cn, digits,
                    "f10_lletra.png")


def main():
    style()
    fig_curves()
    fig_confusion()
    fig_errors()
    fig_ablation()
    fig_data()
    fig_robustness()
    fig_noise()
    fig_weights()
    fig_weights2()
    if os.path.exists(os.path.join(RES, "preprocess_test.npz")):
        fig_preprocess()
    if os.path.exists(os.path.join(RES, "lletra.npz")):
        fig_handwriting()


if __name__ == "__main__":
    main()
