"""Passa els dígits dibuixats a mà pels models i desa els resultats."""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "resultats")
MOD = os.path.join(ROOT, "models")
HANDWRITING = os.path.join(ROOT, "data", "lletra")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessament import preprocess
from xarxa import Network


def main():
    file_paths = [(d, os.path.join(HANDWRITING, f"lletra_{d}.png")) for d in range(10)]
    file_paths = [(d, p) for d, p in file_paths if os.path.exists(p)]
    if not file_paths:
        print("no hi ha cap dibuix a data/lletra/")
        return

    networks = {
        "16": Network.load(os.path.join(MOD, "mlp_relu.npz")),
        "256": Network.load(os.path.join(MOD, "mlp_256_128.npz")),
    }

    table_rows = []
    raw_images, clean_images = [], []
    for d, ruta in file_paths:
        raw_img, clean_img = preprocess(ruta)
        raw_images.append(raw_img)
        clean_images.append(clean_img)
        fila = {"digito": d}
        for name, xarxa in networks.items():
            pc = xarxa.predict_proba(raw_img.reshape(1, -1))[0]
            pn = xarxa.predict_proba(clean_img.reshape(1, -1))[0]
            fila[f"cru_{name}"] = int(pc.argmax())
            fila[f"conf_cru_{name}"] = round(float(pc.max()), 4)
            fila[f"net_{name}"] = int(pn.argmax())
            fila[f"conf_net_{name}"] = round(float(pn.max()), 4)
        table_rows.append(fila)

    field_names = ["digito"]
    for name in networks:
        field_names += [f"cru_{name}", f"conf_cru_{name}",
                   f"net_{name}", f"conf_net_{name}"]
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "lletra.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_names)
        w.writeheader()
        w.writerows(table_rows)

    np.savez_compressed(os.path.join(RESULTS, "lletra.npz"),
                        digits=np.array([d for d, _ in file_paths]),
                        raw_images=np.stack(raw_images), clean_images=np.stack(clean_images))

    n = len(table_rows)
    print(f"{n} dígits")
    for name in networks:
        ok_raw = sum(f[f"cru_{name}"] == f["digito"] for f in table_rows)
        ok_net = sum(f[f"net_{name}"] == f["digito"] for f in table_rows)
        print(f"xarxa {name}: cru {ok_raw}/{n}   net {ok_net}/{n}")
        for f in table_rows:
            print(f"  {f['digito']}: cru {f[f'cru_{name}']} "
                  f"({f[f'conf_cru_{name}'] * 100:.1f}%)  net {f[f'net_{name}']} "
                  f"({f[f'conf_net_{name}'] * 100:.1f}%)")


if __name__ == "__main__":
    main()
