"""Pseudo-verdad de PA, MT y FL a partir de las mascaras de entrenamiento.

El reto NO publica etiquetas de PA/FL/MT: solo mascaras de aponeurosis (1048 imagenes) y
de fasciculo (2761), y ninguna imagen tiene las dos. Para poder validar en local sin mirar
el leaderboard se reconstruye la medida siguiendo el procedimiento de anotacion descrito
en las bases:
  - mascara de aponeurosis -> MT (tres posiciones) y la inclinacion de la profunda;
  - mascara de fasciculo  -> orientacion del fasciculo -> PA respecto a esa inclinacion;
  - FL = MT / sin(PA) (la extrapolacion lineal que usan los anotadores).
La escala fisica se mide con roi_scale sobre la imagen distribuida y se traslada al marco
de la mascara, que conserva el tamano nativo del fotograma.
"""
from __future__ import annotations

import glob
import os

import numpy as np

import geometria as G
import roi_scale as S

RAW = "E:/UMUD-data/raw"


def mask_scale(path_img, mask_shape, cache={}):
    """px/mm en el marco de la mascara (tamano nativo) a partir de la imagen distribuida."""
    if not cache:
        import pandas as pd
        d = pd.read_csv("E:/UMUD-work/scale_all.csv")
        d["key"] = d.path.str.replace("\\", "/", regex=False)
        cache.update(d.set_index("key").to_dict("index"))
    r = cache[str(path_img).replace("\\", "/")]
    Hm, Wm = mask_shape
    sx = r["sx"] * Wm / r["W"] if np.isfinite(r["sx"]) else np.nan
    sy = r["sy"] * Hm / r["H"] if np.isfinite(r["sy"]) else np.nan
    return sx, sy, r


def apo_record(name):
    img = f"{RAW}/apo_imgs_v1/apo_images_new_model_v1/{name}"
    msk = f"{RAW}/apo_masks_v1/apo_masks_new_model_v1/{name}"
    m = S.load_gray(msk)
    m = (m > m.max() * 0.5).astype(np.float32) if m.max() > 0 else m
    sx, sy, r = mask_scale(img, m.shape)
    cur = G.aponeurosis_curves(m)
    mt, sup, deep = G.thickness_mm(cur, sx, sy) if np.isfinite(sx) else (np.nan, None, None)
    slope = G.deep_slope_deg(deep[0], 0.5 * (deep[1] + deep[2]), sx, sy) if deep is not None and np.isfinite(sx) else np.nan
    return dict(name=name, kind="apo", Hm=m.shape[0], Wm=m.shape[1], sx=sx, sy=sy,
                n_curves=len(cur), mt_mm=mt, deep_slope_deg=slope,
                tick_y=r["tick_y"], depth_mm=r["depth_mm"])


def fasc_record(name):
    img = f"{RAW}/fasc_imgs_v1/fasc_images_new_model_v1/{name}"
    msk = f"{RAW}/fasc_masks_v1/fasc_masks_new_model_v1/{name}"
    m = S.load_gray(msk)
    m = (m > m.max() * 0.5).astype(np.float32) if m.max() > 0 else m
    sx, sy, r = mask_scale(img, m.shape)
    th = G.dominant_orientation_deg(m)
    return dict(name=name, kind="fasc", Hm=m.shape[0], Wm=m.shape[1], sx=sx, sy=sy,
                theta_px=th, theta_mm=G.fascicle_angle_deg(th, sx, sy) if np.isfinite(sx) else np.nan,
                frac=float(m.mean()), tick_y=r["tick_y"], depth_mm=r["depth_mm"])


if __name__ == "__main__":
    import sys
    import pandas as pd
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    apo = sorted(os.path.basename(p) for p in glob.glob(f"{RAW}/apo_masks_v1/apo_masks_new_model_v1/*.tif"))[:n]
    fas = sorted(os.path.basename(p) for p in glob.glob(f"{RAW}/fasc_masks_v1/fasc_masks_new_model_v1/*.tif"))[:n]
    da = pd.DataFrame([apo_record(x) for x in apo])
    df = pd.DataFrame([fasc_record(x) for x in fas])
    da.to_csv("E:/UMUD-work/pseudo_apo.csv", index=False)
    df.to_csv("E:/UMUD-work/pseudo_fasc.csv", index=False)
    print("APO n=%d  escala ok=%.2f  curvas>=2=%.2f  MT mm:" % (len(da), da.sy.notna().mean(), (da.n_curves >= 2).mean()))
    print(da.mt_mm.describe().round(2).to_string())
    print("pendiente profunda (grados):", da.deep_slope_deg.describe().round(2).to_dict())
    print("\nFASC n=%d  escala ok=%.2f" % (len(df), df.sy.notna().mean()))
    print(df.theta_mm.describe().round(2).to_string())
