"""Calcula el corte espectral por eje para muchas imagenes y lo guarda, para no releerlas.

Ventana CUADRADA centrada en la regla: el mismo numero de muestras en los dos ejes, porque
medir el corte de x con 23 muestras y el de y con 522 no es comparable y sesga todo (fue mi
primer error aqui). El resultado se guarda en nitidez.csv y el analisis de umbrales se hace
despues sobre el CSV, en segundos.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import nitidez_eje as N
import roi_scale as R

UMUD_WORK = os.environ.get("UMUD_WORK", "E:/UMUD-work")
L = 96
POR_GRUPO = int(sys.argv[1]) if len(sys.argv) > 1 else 150
OUT = f"{UMUD_WORK}/nitidez.csv"
PAT = {"apo": "apo_imgs_v1/apo_images_new_model_v1", "fasc": "fasc_imgs_v1/fasc_images_new_model_v1"}

meta = pd.read_csv(f"{UMUD_WORK}/ds/meta.csv")
tr = meta[meta.kind.isin(["apo", "fasc"]) & meta.Wm.notna()].copy()
tr["aniso"] = (tr.Wi / tr.Wm) / (tr.Hi / tr.Hm)
tr["grupo"] = list(zip(tr.Wm.astype(int), tr.Hm.astype(int)))
sub = tr.groupby("grupo", group_keys=False).head(POR_GRUPO)
filas = []
for i, (_, r) in enumerate(sub.iterrows()):
    f = glob.glob(f"{os.environ.get('UMUD_RAW','E:/UMUD-data/raw')}/{PAT[r['kind']]}/{r['name']}.*")
    if not f:
        continue
    try:
        g = R.load_gray(f[0])
        ry = R.scan_ruler_multi(g, "y")
    except Exception:
        continue
    if not ry:
        continue
    H, W = g.shape
    cxm = int(sum(ry["band"]) / 2); cym = int((ry["first"] + ry["last"]) / 2)
    y0 = max(0, min(H - L, cym - L // 2)); x0 = max(0, min(W - L, cxm - L // 2))
    a = g[y0:y0 + L, x0:x0 + L]
    if a.shape != (L, L):
        continue
    est, cx, cy = N.anisotropia(a)
    if not np.isfinite(est):
        continue
    filas.append(dict(kind=r["kind"], name=r["name"], grupo=str(r.grupo), real=r.aniso,
                      est=est, cx=cx, cy=cy))
    if i % 200 == 0:
        print(i, len(sub), flush=True)
d = pd.DataFrame(filas)
d.to_csv(OUT, index=False)
print("TOTAL", len(d), "->", OUT, flush=True)
print(d.groupby("grupo").agg(n=("est", "size"), real=("real", "median"), est=("est", "median")).round(3).to_string())
