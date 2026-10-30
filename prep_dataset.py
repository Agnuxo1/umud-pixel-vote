"""Prepara un conjunto compacto para entrenar en server-ia.

La imagen distribuida esta reescalada y la mascara conserva el tamano nativo: se llevan
las dos a una rejilla comun de trabajo y se guarda la razon de anisotropia, que hace falta
despues para convertir angulos y distancias a unidades fisicas.

Salida: PNG uint8 de 384x256 (imagen, mascara) + meta.csv. Pesa ~50 veces menos que el
original, asi que se puede subir al servidor sin mover el conjunto completo del concurso.
NO se redistribuyen los datos del reto fuera del entorno de trabajo propio.
"""
from __future__ import annotations

import glob
import os

import numpy as np

import roi_scale as S

_SC = None


def scale_of(kind, name):
    """Lee la escala de la pasada unica (scale_all.csv) en vez de volver a medirla."""
    global _SC
    if _SC is None:
        import pandas as pd
        d = pd.read_csv("E:/UMUD-work/scale_all.csv")
        _SC = {(r["kind"], str(r["name"])): r for _, r in d.iterrows()}
    return _SC[(kind, name)]

RAW = "E:/UMUD-data/raw"
OUT = "E:/UMUD-work/ds"
W, H = 384, 256


def one(kind, name):
    import cv2
    sub = {"apo": ("apo_imgs_v1/apo_images_new_model_v1", "apo_masks_v1/apo_masks_new_model_v1"),
           "fasc": ("fasc_imgs_v1/fasc_images_new_model_v1", "fasc_masks_v1/fasc_masks_new_model_v1")}[kind]
    gi = S.load_gray(f"{RAW}/{sub[0]}/{name}")
    gm = S.load_gray(f"{RAW}/{sub[1]}/{name}")
    Hi, Wi = gi.shape
    Hm, Wm = gm.shape
    stem0 = os.path.splitext(name)[0]
    r = scale_of(kind, stem0)
    im = cv2.resize(gi, (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8)
    # umbral bajo al reducir: interesa conservar la linea, no su grosor exacto
    mk = (cv2.resize(gm, (W, H), interpolation=cv2.INTER_AREA) > gm.max() * 0.25).astype(np.uint8) * 255
    stem = os.path.splitext(name)[0]
    cv2.imwrite(f"{OUT}/{kind}/img/{stem}.png", im)
    cv2.imwrite(f"{OUT}/{kind}/msk/{stem}.png", mk)
    # px/mm en la rejilla de trabajo (imagen distribuida -> WxH)
    sx = r["sx"] * W / Wi if np.isfinite(r["sx"]) else np.nan
    sy = r["sy"] * H / Hi if np.isfinite(r["sy"]) else np.nan
    return dict(kind=kind, name=stem, Wi=Wi, Hi=Hi, Wm=Wm, Hm=Hm,
                aniso_nativa=(Wi / Wm) / (Hi / Hm) if Wm and Hm else np.nan,
                sx=sx, sy=sy, sx_img=r["sx"], sy_img=r["sy"], tick_y=r["tick_y"],
                cov_y=r["cov_y"], band_y=r["band_y"], span_y=r["span_y"])


def test_one(path):
    import cv2
    g = S.load_gray(path)
    Hi, Wi = g.shape
    r = scale_of("test", os.path.splitext(os.path.basename(path))[0])
    stem = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(f"{OUT}/test/img/{stem}.png", cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8))
    return dict(kind="test", name=stem, Wi=Wi, Hi=Hi, Wm=np.nan, Hm=np.nan, aniso_nativa=np.nan,
                sx=r["sx"] * W / Wi if np.isfinite(r["sx"]) else np.nan,
                sy=r["sy"] * H / Hi if np.isfinite(r["sy"]) else np.nan,
                sx_img=r["sx"], sy_img=r["sy"], tick_y=r["tick_y"],
                cov_y=r["cov_y"], band_y=r["band_y"], span_y=r["span_y"])


if __name__ == "__main__":
    import pandas as pd
    for k in ("apo", "fasc"):
        for s in ("img", "msk"):
            os.makedirs(f"{OUT}/{k}/{s}", exist_ok=True)
    os.makedirs(f"{OUT}/test/img", exist_ok=True)
    rows = []
    for k, d in (("apo", "apo_masks_v1/apo_masks_new_model_v1"), ("fasc", "fasc_masks_v1/fasc_masks_new_model_v1")):
        names = sorted(os.path.basename(p) for p in glob.glob(f"{RAW}/{d}/*.tif"))
        for i, n in enumerate(names):
            rows.append(one(k, n))
            if i % 400 == 0:
                print(k, i, flush=True)
    for p in sorted(glob.glob(f"{RAW}/test_images_v2/test_set_v2/*")):
        if p.lower().endswith((".tif", ".png")):
            rows.append(test_one(p))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/meta.csv", index=False)
    print("filas", len(df))
    print(df.groupby("kind").agg(n=("name", "size"), escala=("sy", lambda s: float(s.notna().mean()))).round(3).to_string())
    print("anisotropia nativa (solo entrenamiento):")
    print(df.aniso_nativa.describe().round(3).to_string())
