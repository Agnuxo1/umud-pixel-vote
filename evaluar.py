"""SUPERADO por `evaluar2.py`. Se conserva publicado, no se usa.

Esta es la version 1 del evaluador. Tiene el defecto documentado en RESULTS.md 5.4 (G-06): una
media envenenada por NaN dejaba el angulo sin medir en cuatro de los cinco pliegues. Ninguna cifra
publicada sale de aqui; todas salen de `evaluar2.py`. Queda en el repo porque el defecto y su
correccion son parte del registro, no para ejecutarse.

Evaluacion local honesta de un modelo, sin tocar el leaderboard.

El reto no publica PA/FL/MT, asi que no se puede calcular la metrica oficial en local. Lo
que SI se puede medir, cada cosa donde existe su etiqueta:

  MT  -> en las imagenes con mascara de APONEUROSIS: el grosor que deduce el modelo frente
         al que dan las mascaras, en milimetros. Error directamente comparable con tau=3 mm.
  PA  -> en las imagenes con mascara de FASCICULO: la orientacion que deduce el modelo
         frente a la de la mascara, en grados. Es un limite inferior del error de PA, porque
         PA ademas resta la inclinacion de la aponeurosis profunda. Comparable con tau=6 deg.
  FL  -> NO se puede medir: ninguna imagen tiene las dos mascaras. Se PROPAGA el error desde
         MT y PA por FL = MT/sin(PA), y se marca como propagado, nunca como medido.

Se informa `score_proxy` = (MAE_PA/6 + MAE_FL_propagado/12 + MAE_MT/3)/3, que imita la forma
de la metrica oficial pero NO es la metrica oficial. La decision entre arquitecturas se toma
con esto y con el IoU del grupo de dispositivo retenido, no con el leaderboard.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import geometria as G
import imutil as U
from infer import BIN_CENTERS, angulo_circular, votos
from train import K_ORI, VOCAB, apo_classes, orientation_bins


def mt_de_mascara(msk, sx, sy):
    cur = G.aponeurosis_curves((msk > 0).astype(np.float32), thr=0.5)
    if len(cur) < 2:
        return np.nan
    mt, _, _ = G.thickness_mm([cur[0], cur[-1]], sx, sy)
    return mt


def mt_de_modelo(p, sx, sy):
    cur_s = G.aponeurosis_curves(p[1], thr=0.35)
    cur_d = G.aponeurosis_curves(p[2], thr=0.35)
    if not cur_s or not cur_d:
        cur = G.aponeurosis_curves(p[1] + p[2], thr=0.35)
        if len(cur) < 2:
            return np.nan
        cur_s, cur_d = [cur[0]], [cur[-1]]
    s = max(cur_s, key=lambda t: t[3]); d = max(cur_d, key=lambda t: t[3])
    mt, _, _ = G.thickness_mm([s, d], sx, sy)
    return mt


def ang_de_mascara(msk):
    return G.dominant_orientation_deg((msk > 0).astype(np.float32))


def ang_de_modelo(p):
    pesos = [float(p[3 + k].sum()) for k in range(K_ORI)]
    return angulo_circular(pesos)


def dif_angular(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ds")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    import modelos
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"steps": ck.get("steps", 12)} if ck["arch"] in ("neuropixel", "hybrid") else {}
    model = modelos.build(ck["arch"], ck["vocab"], **kw)
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    meta = pd.read_csv(f"{a.root}/meta.csv")
    meta["grupo"] = meta.Wm.astype(str) + "x" + meta.Hm.astype(str)
    va = meta[(meta.grupo == str(ck["val_group"])) & meta.kind.isin(["apo", "fasc"])]
    if a.max:
        va = va.head(a.max)
    e_mt, e_pa, mts = [], [], []
    for _, r in va.iterrows():
        img = U.imread_gray(f"{a.root}/{r['kind']}/img/{r['name']}.png")
        msk = U.imread_gray(f"{a.root}/{r['kind']}/msk/{r['name']}.png")
        p = votos(model, img, dev)
        if r["kind"] == "apo":
            t = mt_de_mascara(msk, r["sx"], r["sy"]); q = mt_de_modelo(p, r["sx"], r["sy"])
            if np.isfinite(t) and np.isfinite(q):
                e_mt.append(abs(q - t)); mts.append(t)
        else:
            t = ang_de_mascara(msk); q = ang_de_modelo(p)
            if np.isfinite(t) and np.isfinite(q):
                e_pa.append(dif_angular(G.fascicle_angle_deg(q, r["sx"], r["sy"]),
                                        G.fascicle_angle_deg(t, r["sx"], r["sy"])))
    mae_mt = float(np.mean(e_mt)) if e_mt else float("nan")
    mae_pa = float(np.mean(e_pa)) if e_pa else float("nan")
    # propagacion de FL = MT/sin(PA): dFL = dMT/sin(PA) + MT*cos(PA)/sin^2(PA) * dPA(rad)
    mt0 = float(np.median(mts)) if mts else 20.0
    pa0 = 20.0
    s, c = np.sin(np.radians(pa0)), np.cos(np.radians(pa0))
    mae_fl = mae_mt / s + mt0 * c / (s * s) * np.radians(mae_pa)
    proxy = (mae_pa / 6.0 + mae_fl / 12.0 + mae_mt / 3.0) / 3.0
    res = dict(arch=ck["arch"], seed=ck["seed"], val_group=str(ck["val_group"]),
               n_apo=len(e_mt), n_fasc=len(e_pa), mae_mt_mm=mae_mt, mae_pa_deg=mae_pa,
               mae_fl_mm_propagado=float(mae_fl), score_proxy=float(proxy),
               mt_mediana_mascara=mt0, params=modelos.n_params(model))
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
