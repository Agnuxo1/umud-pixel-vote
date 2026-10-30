"""Validacion dura del modulo de escala contra una verdad conocida.

Para las imagenes de ENTRENAMIENTO conocemos el tamano nativo del fotograma, porque la
mascara lo conserva mientras la imagen distribuida esta reescalada. Si el pixel nativo es
isotropo, la anisotropia real del archivo distribuido es

    aniso_real = (Wi / Wm) / (Hi / Hm)

y el modulo de escala, que no ve la mascara, deberia reproducirla con sx/sy. Esto comprueba
la parte del metodo que mas pesa en la metrica, sin tocar el leaderboard.
"""
import glob, os

import numpy as np
import pandas as pd

import roi_scale as R

import os

UMUD_RAW = os.environ.get("UMUD_RAW", "E:/UMUD-data/raw")
UMUD_WORK = os.environ.get("UMUD_WORK", "E:/UMUD-work")

RAW = UMUD_RAW


def tamanos(d):
    import cv2
    out = {}
    for f in glob.glob(f"{RAW}/{d}/*.tif"):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        out[os.path.splitext(os.path.basename(f))[0]] = (im.shape[1], im.shape[0])
    return out


if __name__ == "__main__":
    sc = pd.read_csv(f"{UMUD_WORK}/scale_all.csv")
    am = tamanos("apo_masks_v1/apo_masks_new_model_v1")
    fm = tamanos("fasc_masks_v1/fasc_masks_new_model_v1")
    rows = []
    for _, r in sc.iterrows():
        d = am if r["kind"] == "apo" else (fm if r["kind"] == "fasc" else None)
        if d is None or r["name"] not in d:
            continue
        Wm, Hm = d[r["name"]]
        rows.append(dict(kind=r["kind"], name=r["name"], W=r["W"], H=r["H"], Wm=Wm, Hm=Hm,
                         aniso_real=(r["W"] / Wm) / (r["H"] / Hm),
                         sx=r["sx"], sy=r["sy"], tick_y=r["tick_y"], tick_x=r["tick_x"],
                         cov_y=r["cov_y"], cov_x=r["cov_x"]))
    d = pd.DataFrame(rows)
    d["aniso_med"] = d.sx / d.sy
    fiable = (d.tick_y == d.tick_x) & d.sx.notna() & d.sy.notna() & (d.cov_y == 1.0) & (d.cov_x == 1.0)
    f = d[fiable]
    print(f"imagenes de entrenamiento: {len(d)}  con las dos reglas fiables: {len(f)} ({len(f)/max(len(d),1):.1%})")
    if len(f):
        err = (f.aniso_med / f.aniso_real - 1.0).abs()
        print("error relativo de la anisotropia medida frente a la real:")
        print(err.describe().round(4).to_string())
        print(f"dentro del 5 %: {(err <= 0.05).mean():.1%}   dentro del 10 %: {(err <= 0.10).mean():.1%}")
        print("\npor formato (W x H de la imagen distribuida):")
        print(f.assign(e=err).groupby([f.W, f.H]).agg(n=("e", "size"), real=("aniso_real", "median"),
                                                      medida=("aniso_med", "median"), err=("e", "median")).round(3).to_string())
    d.to_csv(f"{UMUD_WORK}/validacion_escala.csv", index=False)
