"""Executa el pipeline sencer en ordre, sense la finestra de dibuixar."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")


def run(name):
    print(f"\n=== {name} ===", flush=True)
    subprocess.run([sys.executable, os.path.join(SRC, name)], check=True, cwd=ROOT)


if __name__ == "__main__":
    run("mnist.py")
    run("comprovacio_gradient.py")
    run("experiments.py")
    run("figures.py")

    letra = os.path.join(ROOT, "data", "lletra")
    if os.path.isdir(letra) and any(n.endswith(".png") for n in os.listdir(letra)):
        run("lletra.py")

    print("\nFet. Resultats a resultados/, models a modelos/ i figures a figuras/.")
