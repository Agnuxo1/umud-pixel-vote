"""Corrige el valor de marca de un grupo cuando su MT medida es fisiologicamente imposible.

EL FALLO. `TICK_CANDIDATES_MM` era (2, 5, 10) y las reglas de ecografo tambien usan marcas
de 1 mm. Peor: cuando dos candidatos dejan la profundidad dentro del rango plausible
(20-110 mm), `choose_tick_mm` desempata acercando la profundidad al centro del rango, que es
65 mm, un criterio arbitrario que favorece sistematicamente a la marca mas grande. Con una
marca el doble de lo real, `sy` sale a la mitad y todo lo que se mida en milimetros sale al
doble.

COMO SE DETECTA. La mascara de aponeurosis da el grosor muscular en PIXELES, que no se
discute. Convertido con `sy` tiene que caer en el rango fisiologico de estos musculos, de
unos 8 a 45 mm. Medido sobre los seis grupos con mascaras suficientes:

    660x556   18,4 mm      1234x688  19,3 mm      800x652  20,5 mm
    1152x864  23,2 mm      800x600   38,6 mm      512x512  55,6 mm  <- imposible

Los dos grupos altos son exactamente los dos con marca de 2 mm, y dividir su MT por dos los
deja en 27,8 y 19,3 mm, en linea con los otros cuatro. Es la firma de una marca que vale 1 mm
leida como 2.

QUE CORRIGE Y QUE NO. Esto usa las mascaras de ENTRENAMIENTO, asi que corrige la verdad de
terreno con la que se evalua, y hay que corregirla: informar "MAE de 9,9 mm" contra una verdad
que esta al doble es informar mal. **No es aplicable al test**, donde no hay mascaras, y la
ambiguedad de la marca no se puede resolver con la profundidad porque los equipos usan
profundidades genuinamente distintas, de 40 a 100 mm. Se declara como limitacion.

Y no toca la ablacion: la verdad y la prediccion de un pliegue usan el MISMO `sy`, asi que el
factor escala las dos por igual y el orden de las arquitecturas dentro de un pliegue no se
mueve. Lo que cambia es la magnitud absoluta que se publica y la comparacion entre pliegues.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

import geometria as G
import imutil as U

UMUD_WORK = os.environ.get("UMUD_WORK", "E:/UMUD-work")
META = f"{UMUD_WORK}/ds/meta.csv"
MT_LO, MT_HI = 8.0, 45.0
CANDIDATOS = (1.0, 2.0, 5.0, 10.0)
N_MUESTRA = 40


def mt_mediana(meta, grupo, root, n=N_MUESTRA):
    """MT mediana de las mascaras del grupo, en mm y en px."""
    s = meta[(meta.kind == "apo") & (meta.g == grupo) & meta.sx.notna() & meta.sy.notna()].head(n)
    mm, px = [], []
    for _, r in s.iterrows():
        try:
            msk = U.imread_gray(f"{root}/apo/msk/{r['name']}.png")
        except Exception:
            continue
        cur = G.aponeurosis_curves((msk > 0).astype(np.float32), thr=0.5)
        if len(cur) < 2:
            continue
        a, _, _ = G.thickness_mm([cur[0], cur[-1]], r["sx"], r["sy"])
        b, _, _ = G.thickness_mm([cur[0], cur[-1]], 1.0, 1.0)
        if np.isfinite(a):
            mm.append(a); px.append(b)
    if len(mm) < 5:
        return np.nan, np.nan, len(mm)
    return float(np.median(mm)), float(np.median(px)), len(mm)


def main():
    root = f"{UMUD_WORK}/ds"
    meta = pd.read_csv(META)
    meta["g"] = meta.Wm.astype(str) + "x" + meta.Hm.astype(str)
    cambios = {}
    print(f"{'grupo':16s} {'tick':>5s} {'MT_mm':>7s} {'MT_px':>7s} {'tick_nuevo':>10s} {'MT_nueva':>9s}")
    for grupo in sorted(meta[meta.kind == "apo"].g.unique()):
        mm, px, n = mt_mediana(meta, grupo, root)
        if not np.isfinite(mm):
            continue
        t0 = float(meta[(meta.g == grupo) & (meta.kind == "apo")].tick_y.median())
        if MT_LO <= mm <= MT_HI:
            print(f"{grupo:16s} {t0:5.1f} {mm:7.2f} {px:7.1f} {'-':>10s} {'-':>9s}")
            continue
        # la marca es la magnitud ambigua: probar los candidatos y quedarse con el que mete
        # la MT en rango y mas cerca de la mediana de los grupos coherentes
        mejor, mejor_mm, mejor_pen = np.nan, np.nan, np.inf
        for t in CANDIDATOS:
            if not np.isfinite(t0) or t0 <= 0:
                continue
            m2 = mm * t / t0                      # sy ~ 1/t, luego MT ~ t
            pen = (0.0 if MT_LO <= m2 <= MT_HI else 1e3) + abs(m2 - 21.0)
            if pen < mejor_pen:
                mejor, mejor_mm, mejor_pen = t, m2, pen
        print(f"{grupo:16s} {t0:5.1f} {mm:7.2f} {px:7.1f} {mejor:10.1f} {mejor_mm:9.2f}")
        if np.isfinite(mejor) and mejor != t0:
            cambios[grupo] = (t0, mejor, mejor / t0)

    if not cambios:
        print("\nsin cambios: todos los grupos con mascaras dan MT fisiologica")
        return 0
    print("\ncambios de marca (y por tanto de escala):")
    for g_, (a, b, f) in cambios.items():
        print(f"  {g_}: marca {a} -> {b} mm, sy x{1/f:.2f}, todo lo que se mide en mm x{f:.2f}")
    if "--aplicar" not in sys.argv:
        print("\n(no se ha escrito nada; pasar --aplicar para corregir meta.csv)")
        return 0
    shutil.copy(META, META + ".antes_de_la_marca")
    for g_, (a, b, f) in cambios.items():
        m = meta.g == g_
        for c in ("sy", "sx", "sy_img", "sx_img"):
            if c in meta.columns:
                meta.loc[m, c] = meta.loc[m, c] / f      # sy ~ 1/tick
        meta.loc[m, "tick_y"] = b
    meta.drop(columns=["g"]).to_csv(META, index=False)
    print(f"\nescrito {META} (respaldo en {os.path.basename(META)}.antes_de_la_marca)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
