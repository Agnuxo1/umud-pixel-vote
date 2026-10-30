"""Operaciones de imagen con numpy/scipy/PIL, sin OpenCV ni scikit-image.

server-ia trae torch, numpy, scipy, pandas, PIL, torchvision, sklearn y matplotlib, pero no
cv2 ni skimage. El codigo que corre en el servidor (entrenamiento e inferencia) usa solo
este modulo, de modo que el entorno reproducible es el minimo de `requirements.txt` y no
hace falta instalar nada en el trabajo de cola.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def imread_gray(path):
    from PIL import Image
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    return np.asarray(im)


def sobel_xy(a):
    a = a.astype(np.float32)
    return ndi.sobel(a, axis=1, mode="nearest"), ndi.sobel(a, axis=0, mode="nearest")


def gauss(a, sigma):
    return ndi.gaussian_filter(a.astype(np.float32), sigma, mode="nearest")


def close_rect(bw, h, w):
    return ndi.binary_closing(bw.astype(bool), structure=np.ones((h, w), bool))


def open_rect(bw, h, w):
    return ndi.binary_opening(bw.astype(bool), structure=np.ones((h, w), bool))


def components(bw):
    """Devuelve (etiquetas, lista de dicts con bbox, area y centroide)."""
    lab, n = ndi.label(bw.astype(bool))
    if n == 0:
        return lab, []
    objs = ndi.find_objects(lab)
    areas = ndi.sum_labels(np.ones_like(lab, np.float32), lab, index=np.arange(1, n + 1))
    cen = ndi.center_of_mass(np.ones_like(lab, np.float32), lab, index=np.arange(1, n + 1))
    out = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        out.append(dict(label=i + 1, y0=sl[0].start, y1=sl[0].stop, x0=sl[1].start, x1=sl[1].stop,
                        w=sl[1].stop - sl[1].start, h=sl[0].stop - sl[0].start,
                        area=float(areas[i]), cy=float(cen[i][0]), cx=float(cen[i][1])))
    return lab, out


def radon_variance(a, thetas_deg):
    """Varianza de la proyeccion para cada angulo (sustituto de skimage.transform.radon).

    Se gira la imagen y se suma por columnas: el angulo cuya proyeccion es mas picuda
    corresponde a la orientacion dominante de las lineas.
    """
    a = a.astype(np.float32)
    a = a - a.mean()
    out = []
    for t in thetas_deg:
        r = ndi.rotate(a, t, reshape=False, order=1, mode="constant", cval=0.0)
        out.append(float(r.sum(axis=0).var()))
    return np.array(out)
