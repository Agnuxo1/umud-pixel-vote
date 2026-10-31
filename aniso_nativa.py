"""Escala por huella de equipo, con la anisotropia aprendida del tamano del fotograma.

DE DONDE SALE LA INFORMACION. La mascara de entrenamiento conserva el tamano nativo del
fotograma mientras la imagen distribuida esta reescalada, asi que para esas imagenes se
conoce la anisotropia del archivo distribuido:

    aniso = (Wi / Wm) / (Hi / Hm)

Eso NO es una etiqueta del reto: no son PA, FL ni MT, es el tamano de un fichero. Aprender
de ahi la anisotropia de cada equipo y aplicarla al test es inferencia normal, no fuga. Se
declara porque cambia lo que se puede afirmar: las imagenes de test cuyo equipo no aparece
en entrenamiento no reciben nada y se quedan en isotropia.

POR QUE HACE FALTA. La lectura directa de las dos reglas es excelente cuando existe (error
mediano de la razon 0,3 %), pero solo el 38 % del entrenamiento y el 12 % del test tienen
las dos reglas legibles. Donde hay una sola, el modulo devuelve sx = sy, y esa poblacion no
es isotropa: su error mediano es del 11,1 %, que es exactamente 0,125/1,125, el caso de un
fotograma 1152x864 servido como 1200x800.

POR QUE LA HUELLA FUNCIONA. Medido sobre 3809 imagenes: la huella de equipo
(formato, donde esta la regla, tramo de profundidad) separa los fotogramas nativos con un
95,4 % de pureza, y asignar a cada imagen la anisotropia mediana de su huella deja un error
mediano del 0,00 % con el 95,2 % dentro del 5 %.

CASCADA. huella completa -> solo formato (W,H) -> isotropia. El respaldo por formato existe
porque 50 imagenes de test son 1088x644 con una banda que el entrenamiento no tiene, aunque
el formato si. Cada fila registra en `sx_src` de donde salio su anisotropia, para que nadie
tenga que adivinarlo despues.

Una medida propia fiable SIEMPRE manda sobre la de la huella: se corrige lo que no se ha
podido medir, no lo que si.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_N = 5          # filas con anisotropia conocida para fiarse de una huella
ANISO_LO, ANISO_HI = 0.5, 2.0


def huella(df):
    """Formato + donde empieza la banda de la regla + tramo de profundidad, discretizados.

    No se incluye el valor de marca, al contrario que `roi_scale.huella_equipo`: ese valor
    cambia cuando cambia el lector de reglas, y entonces las huellas aprendidas con una
    version del lector no valen para otra. Formato, banda y tramo son propiedades de la
    imagen, no del codigo que la mide.
    """
    b = (df.band_y / 100).round().fillna(-1)
    s = (df.span_y / 25).round().fillna(-1)
    return list(zip(df.W, df.H, b, s))


def aprende(df, excluir_grupos=()):
    """Tablas de anisotropia y de sy por huella y por formato, solo con entrenamiento.

    `excluir_grupos` aparta grupos de dispositivo (Wm, Hm) para poder medir la transferencia
    a un equipo que no se ha visto, que es lo que de verdad pasa en el test.
    """
    tr = df[df.kind.isin(["apo", "fasc"]) & df.Wm.notna() & df.Hm.notna()].copy()
    if excluir_grupos:
        g = list(zip(tr.Wm, tr.Hm))
        tr = tr[[x not in set(excluir_grupos) for x in g]]
    tr["aniso"] = (tr.W / tr.Wm) / (tr.H / tr.Hm)
    tr = tr[np.isfinite(tr.aniso) & (tr.aniso > ANISO_LO) & (tr.aniso < ANISO_HI)]
    tr["huella"] = huella(tr)
    tr["formato"] = list(zip(tr.W, tr.H))
    def med(key, col):
        g = tr.groupby(key)[col]
        m, n = g.median(), g.count()
        return {k: float(v) for k, v in m.items() if n[k] >= MIN_N}
    return dict(aniso_huella=med("huella", "aniso"), aniso_formato=med("formato", "aniso"),
                sy_huella=med("huella", "sy"), sy_formato=med("formato", "sy"))


def aplica(df, tablas):
    """Devuelve una copia con sx, sy, sx_src y sy_src completados por la cascada."""
    d = df.copy()
    d["huella"] = huella(d)
    d["formato"] = list(zip(d.W, d.H))
    # una lectura propia de las dos reglas, con el mismo valor de marca, manda sobre todo
    propia = (d.tick_y == d.tick_x) & d.sx.notna() & d.sy.notna()

    d["sy_src"] = np.where(d.sy.notna(), "regla", "")
    for col, clave in (("huella", "sy_huella"), ("formato", "sy_formato")):
        falta = d.sy.isna()
        if not falta.any():
            break
        v = d.loc[falta, col].map(tablas[clave])
        d.loc[falta, "sy"] = v
        d.loc[falta & v.notna().reindex(d.index, fill_value=False), "sy_src"] = col

    a = pd.Series(np.nan, index=d.index)
    d["sx_src"] = ""
    a_h = d.huella.map(tablas["aniso_huella"]); a_f = d.formato.map(tablas["aniso_formato"])
    a = a_h.where(a_h.notna(), a_f)
    d.loc[a_h.notna(), "sx_src"] = "aniso_huella"
    d.loc[a_h.isna() & a_f.notna(), "sx_src"] = "aniso_formato"
    a = a.fillna(1.0)
    d.loc[a_h.isna() & a_f.isna(), "sx_src"] = "isotropo"

    d.loc[~propia, "sx"] = d.loc[~propia, "sy"] * a[~propia]
    d.loc[propia, "sx_src"] = "regla"
    d["depth_mm"] = d.span_y / d.sy
    return d.drop(columns=["huella", "formato"])
