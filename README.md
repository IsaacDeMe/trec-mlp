# TREC: xarxa neuronal des de zero

Implementació d'un Perceptró Multicapa (MLP) amb NumPy, sense cap framework d'aprenentatge automàtic, per classificar els dígits escrits a mà del conjunt MNIST.

## Estructura

```text
src/
  mnist.py                 baixada i preparació de MNIST
  xarxa.py                 la classe Network, el forward, el backward i l'entrenament
  experiments.py           tots els experiments i resultats
  comprovacio_gradient.py  comprovació numèrica del backward
  preprocessament.py       preprocessament dels dígits propis
  lletra.py                passa els dígits dibuixats a mà pels models
  comparacions.py          regressió logística i kNN
  figures.py               figures del treball
  dibuixa_digits.py        finestra per dibuixar els dígits
executa_tot.py             executa el pipeline sencer
```

## Instal·lació

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Cal Python 3.10 o superior. `tkinter` forma part de la instal·lació estàndard de Python.

## Execució

```bash
python executa_tot.py
```

El pipeline baixa MNIST si no el troba, entrena els models, calcula les mètriques i genera les figures. Els resultats es desen a `resultats/`, els models a `models/` i les figures a `figures/`.

## Dades

Les imatges són de MNIST (LeCun et al., 1998). El codi les baixa del mirall de Google Cloud i es poden consultar a:

- https://github.com/cvdfoundation/mnist
- https://www.tensorflow.org/datasets/catalog/mnist

## Model principal

L'arquitectura del model principal és `[784, 16, 16, 10]`: 784 entrades, dues capes ocultes de 16 neurones i 10 sortides. Té 13.002 paràmetres i s'entrena en pocs minuts en CPU.

## Notes

- Els noms dels fitxers són en català; els identificadors del codi són en anglès.
- Els comentaris `#` s'han eliminat del codi.
- Les carpetes `data/`, `models/`, `resultats/` i `figures/` no es pugen al repositori.
