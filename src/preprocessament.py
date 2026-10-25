"""Preprocessament dels digits propis: retall, escala a 20x20 i centrat pel centre de massa."""

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "resultados")
LETRA = os.path.join(DATA, "lletra")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_image(origen):
    if isinstance(origen, str):
        im = Image.open(origen)
    else:
        im = origen
    im = im.convert("L")
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a


def invert_if_light(a):

    borde = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if borde.mean() > 0.5:
        a = 1.0 - a
    return a


def normalitzar(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def bounding_box(a, umbral=0.3):
    mask = a > umbral
    if not mask.any():
        return None
    table_rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return table_rows[0], table_rows[-1] + 1, cols[0], cols[-1] + 1


def scale_20(recorte):
    h, w = recorte.shape
    scale = 20.0 / max(h, w)
    nuevo = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    im = Image.fromarray((recorte * 255).astype(np.uint8))
    im = im.resize(nuevo, Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def center_mass(a):
    total = a.sum()
    if total < 1e-6:
        return a
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    cy = (ys * a).sum() / total
    cx = (xs * a).sum() / total
    dy = int(round(13.5 - cy))
    dx = int(round(13.5 - cx))
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def a_28(a):
    im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.resize((28, 28), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def preprocess(origen):
    raw_image = invert_if_light(load_image(origen))
    raw_image = a_28(raw_image)

    clean_img = invert_if_light(load_image(origen))
    clean_img = normalitzar(clean_img)
    digit_box = bounding_box(clean_img)
    if digit_box is None:
        return raw_image, np.zeros((28, 28), dtype=np.float32)

    y0, y1, x0, x1 = digit_box
    recorte = clean_img[y0:y1, x0:x1]
    recorte = scale_20(recorte)

    lienzo = np.zeros((28, 28), dtype=np.float32)
    h, w = recorte.shape
    y = (28 - h) // 2
    x = (28 - w) // 2
    lienzo[y:y + h, x:x + w] = recorte
    return raw_image, center_mass(lienzo)


def prueba():
    from mnist import load_cached
    from xarxa import Network

    d = load_cached()
    X, y = d["X_test"], d["y_test"]
    rng = np.random.default_rng(7)

    chosen = [np.where(y == k)[0][0] for k in range(10)]
    raw_images, clean_images = [], []
    for idx in chosen:
        base = Image.fromarray((X[idx].reshape(28, 28) * 255).astype(np.uint8))
        base = base.resize((360, 360), Image.LANCZOS)
        lienzo = Image.new("L", (560, 560), 0)

        base = base.rotate(float(rng.uniform(-8, 8)), resample=Image.BICUBIC,
                           expand=True, fillcolor=0)
        px = int(rng.integers(40, 560 - base.width - 40))
        py = int(rng.integers(40, 560 - base.height - 40))
        lienzo.paste(base, (px, py))
        raw_img, lim = preprocess(lienzo)
        raw_images.append(raw_img)
        clean_images.append(lim)

    raw_images = np.stack(raw_images)
    clean_images = np.stack(clean_images)

    xarxa = Network.load(os.path.join(ROOT, "modelos", "mlp_256_128.npz"))
    p_raw = xarxa.predict_proba(raw_images.reshape(10, -1))
    p_lim = xarxa.predict_proba(clean_images.reshape(10, -1))

    print("dígits:", list(range(10)))
    print("cru:  ", p_raw.argmax(1).tolist(),
          [round(float(v) * 100, 1) for v in p_raw.max(1)])
    print("net:  ", p_lim.argmax(1).tolist(),
          [round(float(v) * 100, 1) for v in p_lim.max(1)])
    print("encerts cru/net:",
          int((p_raw.argmax(1) == np.arange(10)).sum()), "/",
          int((p_lim.argmax(1) == np.arange(10)).sum()))

    os.makedirs(RES, exist_ok=True)
    np.savez_compressed(os.path.join(RES, "preprocess_test.npz"),
                        raw_images=raw_images, clean_images=clean_images,
                        proba_raw=p_raw, proba_net=p_lim)


if __name__ == "__main__":
    prueba()
