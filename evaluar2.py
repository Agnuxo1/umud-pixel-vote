"""Arnes de evaluacion v2. Corrige un fallo real del v1 (evaluar.py).

FALLO DEL v1. `ang_de_modelo` sumaba la probabilidad de cada bin de orientacion sobre
TODA la imagen, fondo incluido. El fondo es ~98 % de los pixeles, asi que esa suma esta
dominada por el sesgo uniforme del fondo y no por los fasciculos: el angulo que salia era
practicamente el mismo para las cuatro arquitecturas y variaba solo con el pliegue
(83 deg / 74 deg / 64 deg), que es la firma de una medida que no depende del modelo.
La ruta real de inferencia (`infer.medir`) NO hace eso: pesa cada bin solo dentro de la
banda entre las dos aponeurosis. El v1 medía algo que el sistema no calcula.

QUE MIDE EL v2. Dos lecturas del mismo cabezal de orientacion, deliberadamente una
pesimista y otra optimista, porque la verdad esta entre las dos:

  ang_libre  -> pesa cada bin por la masa de NO-FONDO que el propio modelo asigna
                (1 - p_fondo). No usa la etiqueta: es lo que el sistema podria hacer en
                test, y es el analogo honesto de la banda `dentro` de infer.medir cuando
                no hay aponeurosis que delimite la banda.
  ang_oracle -> pesa cada bin por la mascara de fasciculo VERDADERA. Usa la etiqueta, asi
                que no se puede presentar nunca como rendimiento del sistema: dice lo buena
                que es la orientacion alli donde de verdad hay fasciculo.

CORRECCION MEDIDA. Esta cabecera decia que `ang_oracle` es un "limite optimista" de
`ang_libre`. No lo es, y lo dice la medida: en el pliegue 1152x864 con 40 imagenes de
fasciculo, `ang_libre` da 1,19 deg y `ang_oracle` 2,32 deg, es decir el oraculo es PEOR. La
razon es un sesgo del protocolo, no del modelo: la referencia es la orientacion dominante de
la mascara ENTERA, o sea una medida global, y `ang_libre` tambien agrega sobre toda la
imagen, mientras el oraculo se restringe a la region de la mascara y mide algo mas local. La
referencia favorece a la lectura que se le parece. Asi que las dos lecturas no estan
ordenadas: son dos medidas distintas, y ninguna acota a la otra.

FL sigue sin poder medirse (ninguna imagen tiene las dos mascaras), pero el v1 lo
propagaba linealizando FL = MT/sin(PA) en torno a PA = 20 deg, y con errores de PA de 80 deg
esa linealizacion no significa nada: daba FL de 300-600 mm, fuera de todo rango
fisiologico (30-200 mm). Aqui se propaga por SIMULACION sobre el rango fisiologico de PA y
se marca como cota, no como medida. Si el error de PA es grande, la cota se declara
no informativa en vez de imprimir un numero absurdo.

Nada de esto es la metrica oficial del reto, que necesita PA/FL/MT verdaderos y no se
publican. `score_proxy` imita su forma para ordenar arquitecturas, nada mas.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import geometria as G
import imutil as U
from infer import BIN_CENTERS, angulo_circular, votos
from train import K_ORI


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


def ang_libre(p):
    """Orientacion pesada por la masa de no-fondo del propio modelo. No usa la etiqueta."""
    w = 1.0 - p[0]
    pesos = [float((p[3 + k] * w).sum()) for k in range(K_ORI)]
    return angulo_circular(pesos)


def ang_oracle(p, msk):
    """Orientacion pesada por la mascara verdadera.

    Es la lectura "oraculo": usa la etiqueta, asi que NO es una capacidad del sistema acabado.
    CORRECCION 7-11-2026: esta linea la llamaba "limite optimista", descripcion retractada en la
    cabecera de este fichero y en RESULTS.md 5.1 porque en los cuatro brazos de 1152x864 la
    lectura oraculo sale PEOR que la libre. Es otra lectura, no una cota."""
    w = (msk > 0).astype(np.float32)
    if w.sum() <= 0:
        return np.nan
    pesos = [float((p[3 + k] * w).sum()) for k in range(K_ORI)]
    return angulo_circular(pesos)


def dif_angular(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def cota_fl(mae_mt, mae_pa, mt0, pa_lo=8.0, pa_hi=35.0, n=64):
    """Cota de |dFL| por simulacion sobre el rango fisiologico de PA, sin linealizar.

    Devuelve (cota, informativa). Con errores de PA comparables al propio rango de PA la
    cota explota y deja de decir nada: se marca no informativa en vez de publicar el numero.
    """
    pa = np.linspace(pa_lo, pa_hi, n)
    s = np.sin(np.radians(pa))
    base = mt0 / s
    peor = (mt0 + mae_mt) / np.sin(np.radians(np.clip(pa - mae_pa, 1.0, 89.0)))
    mejor = (mt0 - mae_mt) / np.sin(np.radians(np.clip(pa + mae_pa, 1.0, 89.0)))
    d = float(np.mean(np.maximum(np.abs(peor - base), np.abs(mejor - base))))
    return d, bool(mae_pa < 0.5 * (pa_hi - pa_lo) and d < 200.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ds")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    import modelos
    # weights_only=True: los checkpoints solo llevan tensores y escalares, y asi un .pt
    # manipulado no puede ejecutar codigo al cargarse.
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"steps": ck.get("steps", 12)} if ck["arch"] in ("neuropixel", "hybrid") else {}
    model = modelos.build(ck["arch"], ck["vocab"], **kw)
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    meta = pd.read_csv(f"{a.root}/meta.csv")
    meta["grupo"] = meta.Wm.astype(str) + "x" + meta.Hm.astype(str)
    va = meta[(meta.grupo == str(ck["val_group"])) & meta.kind.isin(["apo", "fasc"])]
    if a.max:
        # N de CADA tipo, no las N primeras: meta.csv viene ordenado, asi que `head` cogia
        # solo mascaras de aponeurosis y la medida de angulo salia vacia.
        va = pd.concat([g.head(a.max) for _, g in va.groupby("kind")], ignore_index=True)
    e_mt, e_libre, e_oracle, mts = [], [], [], []
    n_sin_escala = 0        # imagenes de fasciculo descartadas por no tener escala
    for _, r in va.iterrows():
        img = U.imread_gray(f"{a.root}/{r['kind']}/img/{r['name']}.png")
        msk = U.imread_gray(f"{a.root}/{r['kind']}/msk/{r['name']}.png")
        p = votos(model, img, dev)
        if r["kind"] == "apo":
            t = mt_de_mascara(msk, r["sx"], r["sy"]); q = mt_de_modelo(p, r["sx"], r["sy"])
            if np.isfinite(t) and np.isfinite(q):
                e_mt.append(abs(q - t)); mts.append(t)
        else:
            t = G.dominant_orientation_deg((msk > 0).astype(np.float32))
            if not np.isfinite(t):
                continue
            tm = G.fascicle_angle_deg(t, r["sx"], r["sy"])
            if not np.isfinite(tm):
                # Sin escala fisica no hay verdad con la que comparar. Antes solo se
                # comprobaba que el angulo del MODELO fuera finito, y la verdad NaN entraba
                # en la lista: un solo valor dejaba la media en NaN, y asi salieron cuatro de
                # los cinco pliegues. Se cuentan los descartes en vez de esconderlos.
                n_sin_escala += 1
                continue
            for fn, acc in ((ang_libre(p), e_libre), (ang_oracle(p, msk), e_oracle)):
                if np.isfinite(fn):
                    acc.append(dif_angular(G.fascicle_angle_deg(fn, r["sx"], r["sy"]), tm))
    def media(xs, nombre):
        """Media que avisa si le han colado un NaN, en vez de devolver NaN en silencio."""
        if not xs:
            return float("nan")
        v = np.asarray(xs, float)
        malos = int((~np.isfinite(v)).sum())
        if malos:
            print(f"AVISO evaluar2: {malos} de {len(v)} valores de {nombre} no son finitos; "
                  f"se excluyen de la media", flush=True)
            v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    mae_mt = media(e_mt, "MT")
    mae_libre = media(e_libre, "angulo libre")
    mae_oracle = media(e_oracle, "angulo oraculo")
    mt0 = float(np.median(mts)) if mts else 20.0
    fl_libre, ok_libre = cota_fl(mae_mt, mae_libre, mt0)
    fl_oracle, ok_oracle = cota_fl(mae_mt, mae_oracle, mt0)
    res = dict(arch=ck["arch"], seed=ck["seed"], val_group=str(ck["val_group"]),
               n_apo=len(e_mt), n_fasc=len(e_oracle), n_fasc_sin_escala=n_sin_escala,
               mae_mt_mm=mae_mt, mt_mediana_mascara=mt0,
               mae_pa_libre_deg=mae_libre, mae_pa_oracle_deg=mae_oracle,
               cota_fl_libre_mm=fl_libre, cota_fl_libre_informativa=ok_libre,
               cota_fl_oracle_mm=fl_oracle, cota_fl_oracle_informativa=ok_oracle,
               params=modelos.n_params(model))
    # proxy solo con la lectura oraculo y solo si su cota de FL dice algo
    if ok_oracle:
        res["score_proxy_oracle"] = float((mae_oracle / 6.0 + fl_oracle / 12.0 + mae_mt / 3.0) / 3.0)
    else:
        res["score_proxy_oracle"] = None
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
