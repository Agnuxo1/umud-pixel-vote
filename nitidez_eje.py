"""Anisotropia por el corte espectral de los graficos que el ecografo sobreimprime.

IDEA. La imagen distribuida es una copia reescalada del fotograma. Ampliar un eje por f
convoluciona ese eje con el nucleo de interpolacion, que recorta el espectro a |v| < 0,5/f:
los trazos de ese eje se vuelven mas romos de forma medible. Los graficos sobreimpresos
(reglas, marcas) son trazos de 1 px nativo, es decir el patron de prueba ideal, con espectro
plano hasta Nyquist antes del reescalado.

POR QUE ES ADIMENSIONAL Y POR ESO PUEDE GENERALIZAR. No se mide la nitidez absoluta, que
depende del equipo, sino la RAZON entre los dos ejes de la MISMA imagen. Si en el fotograma
nativo los trazos son igual de nitidos en horizontal y en vertical — cierto para graficos
sinteticos de un pixel — la razon de cortes da la anisotropia sin calibrar nada por equipo.
Eso es justo lo que hundio a la huella de equipo: necesitaba conocer el equipo (E13b).

JEV eligio este camino con 0,89 de confianza, la mas alta del proyecto, sobre invertir los
picos del nucleo, un regresor espectral y medir los glifos del texto.
"""
from __future__ import annotations

import numpy as np


def corte_espectral(a, eje, q=0.90):
    """Frecuencia normalizada por debajo de la cual queda la fraccion q de la energia.

    Se mide sobre el GRADIENTE en ese eje: el gradiente realza los bordes, que es donde vive
    la huella del nucleo, y elimina el fondo lento del tejido, que no dice nada del
    reescalado. Devuelve NaN si no hay energia suficiente.
    """
    a = np.asarray(a, np.float64)
    if a.ndim != 2 or min(a.shape) < 8:
        return np.nan
    g = np.diff(a, axis=eje)
    if g.size == 0 or not np.isfinite(g).all():
        return np.nan
    n = g.shape[eje]
    if n < 8:
        return np.nan
    w = np.hanning(n)
    w = w.reshape((-1, 1) if eje == 0 else (1, -1))
    P = np.abs(np.fft.rfft(g * w, axis=eje)) ** 2
    P = P.mean(axis=1 - eje)
    P[0] = 0.0
    tot = P.sum()
    if tot <= 0:
        return np.nan
    fr = np.fft.rfftfreq(n)
    acc = np.cumsum(P) / tot
    return float(np.interp(q, acc, fr))


def anisotropia(g, banda=None, q=0.90):
    """Razon de cortes entre ejes -> anisotropia del pixel distribuido.

    Un eje mas ampliado tiene el corte mas bajo, asi que f_x/f_y ~ corte_y/corte_x.
    `banda` acota la region a los graficos sobreimpresos (y0, y1, x0, x1); sin ella se usa la
    imagen entera, donde el moteado del tejido mete su propia anisotropia, que es del equipo
    y no del reescalado.
    """
    a = g.astype(np.float64)
    if banda is not None:
        y0, y1, x0, x1 = banda
        a = a[max(0, y0):y1, max(0, x0):x1]
    cy = corte_espectral(a, 0, q)
    cx = corte_espectral(a, 1, q)
    if not (np.isfinite(cx) and np.isfinite(cy)) or cx <= 0:
        return np.nan, cx, cy
    return float(cy / cx), cx, cy
