"""Finestra per dibuixar els deu dígits amb el ratolí i desar-los com a PNG."""

import os
import sys
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LETRA = os.path.join(ROOT, "data", "lletra")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessament import preprocess

LADO = 560
GROSOR = 28


class Board:
    def __init__(self, ventana):
        self.ventana = ventana
        self.digit = tk.IntVar(value=0)
        self.pil = Image.new("L", (LADO, LADO), 0)
        self.dibujo = ImageDraw.Draw(self.pil)
        self.ultimo = None

        ventana.title("Dibuixa els dígits 0-9")
        ventana.configure(bg="#1a202c")

        izq = tk.Frame(ventana, bg="#1a202c")
        izq.pack(side="left", padx=12, pady=12)

        self.canvas = tk.Canvas(izq, width=LADO, height=LADO, bg="black",
                                highlightthickness=1, highlightbackground="#4a5568",
                                cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.start)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.end)

        der = tk.Frame(ventana, bg="#1a202c")
        der.pack(side="left", fill="y", padx=(0, 14), pady=12)

        tk.Label(der, text="quin dígit?", bg="#1a202c", fg="#e2e8f0",
                 font=("Segoe UI", 11)).pack(anchor="w")

        fila = tk.Frame(der, bg="#1a202c")
        fila.pack(anchor="w", pady=(4, 10))
        self.botones = []
        for d in range(10):
            b = tk.Button(fila, text=str(d), width=2, relief="flat",
                          bg="#2d3748", fg="#e2e8f0", activebackground="#4a5568",
                          command=lambda d=d: self.select(d))
            b.pack(side="left", padx=1)
            self.botones.append(b)

        self.instruccion = tk.Label(der, text="dibuixa el 0", bg="#1a202c",
                                    fg="#f6e05e", font=("Segoe UI", 13, "bold"))
        self.instruccion.pack(anchor="w", pady=(0, 10))

        tk.Label(der, text="vista prèvia 28x28", bg="#1a202c", fg="#a0aec0",
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.preview = tk.Canvas(der, width=140, height=140, bg="black",
                                 highlightthickness=1, highlightbackground="#4a5568")
        self.preview.pack(anchor="w", pady=(3, 12))

        self.aviso = tk.Label(der, text="", bg="#1a202c", fg="#68d391",
                              font=("Segoe UI", 9), wraplength=170, justify="left")
        self.aviso.pack(anchor="w", pady=(0, 8))

        tk.Button(der, text="Desa", width=16, bg="#2b6cb0", fg="white",
                  activebackground="#2c5282", relief="flat",
                  command=self.save_drawing).pack(anchor="w", pady=2)
        tk.Button(der, text="Neteja", width=16, bg="#4a5568", fg="white",
                  activebackground="#2d3748", relief="flat",
                  command=self.clear).pack(anchor="w", pady=2)

        tk.Label(der, text="Enter: desa\nEsc: neteja", bg="#1a202c", fg="#718096",
                 font=("Segoe UI", 8), justify="left").pack(anchor="w", pady=(10, 0))

        ventana.bind("<Return>", lambda e: self.save_drawing())
        ventana.bind("<Escape>", lambda e: self.clear())
        self.select(0)

    def select(self, d):
        self.digit.set(d)
        self.instruccion.config(text=f"dibuixa el {d}")
        for i, b in enumerate(self.botones):
            if i == d:
                b.config(bg="#2b6cb0", fg="white")
            else:
                b.config(bg="#2d3748", fg="#e2e8f0")

    def dot(self, x, y):
        r = GROSOR // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")
        self.dibujo.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def line(self, x0, y0, x1, y1):
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=GROSOR,
                                capstyle=tk.ROUND, smooth=True)
        self.dibujo.line([(x0, y0), (x1, y1)], fill=255, width=GROSOR,
                         joint="curve")
        self.dot(x1, y1)

    def start(self, evento):
        self.ultimo = (evento.x, evento.y)
        self.dot(evento.x, evento.y)

    def drag(self, evento):
        if self.ultimo is None:
            self.ultimo = (evento.x, evento.y)
        self.line(*self.ultimo, evento.x, evento.y)
        self.ultimo = (evento.x, evento.y)

    def end(self, evento):
        self.ultimo = None
        self.update_preview()

    def update_preview(self, clean_img=None):
        if clean_img is None:
            _, clean_img = preprocess(self.pil)
        self.preview.delete("all")
        bloque = 5
        for i in range(28):
            for j in range(28):
                v = int(round(float(clean_img[i, j]) * 255))
                if v > 8:
                    color = f"#{v:02x}{v:02x}{v:02x}"
                    self.preview.create_rectangle(j * bloque, i * bloque,
                                                  (j + 1) * bloque, (i + 1) * bloque,
                                                  fill=color, outline="")

    def save_drawing(self):
        if np.asarray(self.pil).max() == 0:
            self.aviso.config(text="el canvas és buit", fg="#fc8181")
            return
        os.makedirs(LETRA, exist_ok=True)
        d = int(self.digit.get())
        ruta = os.path.join(LETRA, f"lletra_{d}.png")
        self.pil.save(ruta)
        _, clean_img = preprocess(self.pil)
        self.update_preview(clean_img)
        faltan = [k for k in range(10)
                  if not os.path.exists(os.path.join(LETRA, f"lletra_{k}.png"))]
        if faltan:
            self.aviso.config(text=f"desat el {d}. falten: {faltan}", fg="#68d391")
        else:
            self.aviso.config(text="ja hi són tots deu. tanca la finestra.",
                              fg="#68d391")

    def clear(self):
        self.canvas.delete("all")
        self.pil = Image.new("L", (LADO, LADO), 0)
        self.dibujo = ImageDraw.Draw(self.pil)
        self.preview.delete("all")
        self.aviso.config(text="")


def main():
    raiz = tk.Tk()
    Board(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
