"""Inferencia: del campo de votacion por pixel a pa_deg, fl_mm y mt_mm.

La etapa geometrica es IDENTICA para los tres modelos (control, neuropixel, hibrido): lo
unico que cambia es quien produce los votos. Asi la ablacion mide la capa propia y no la
tuberia entera.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

import geometria as G
from train import K_ORI, VOCAB

BIN_W = 180.0 / K_ORI
BIN_CENTERS = -90.0 + BIN_W * (np.arange(K_ORI) + 0.5)


def votos(model, img, dev, tta=True):
    """Probabilidad por pixel de cada palabra. TTA con espejo horizontal."""
    x = torch.from_numpy((img.astype(np.float32) / 255.0 - 0.5)[None, None]).to(dev)
    with torch.no_grad():
        p = model(x).softmax(1)
        if tta:
            q = model(torch.flip(x, [-1])).softmax(1)
            q = torch.flip(q, [-1])
            # al reflejar, el angulo cambia de signo: se invierte el orden de los bins
            q = torch.cat([q[:, :3], torch.flip(q[:, 3:], [1])], 1)
            p = 0.5 * (p + q)
    return p[0].cpu().numpy()


def angulo_circular(pesos):
    """Media de orientaciones en [-90,90): se promedia el doble del angulo (eje, no vector)."""
    w = np.asarray(pesos, float)
    if w.sum() <= 0:
        return np.nan
    a = np.radians(2 * BIN_CENTERS)
    return float(np.degrees(0.5 * np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())))


def medir(p, sx, sy, fallback=(15.0, 70.0, 20.0)):
    """p: V x H x W de probabilidades. sx, sy: px/mm en esa rejilla."""
    sup, deep = p[1], p[2]
    ori = p[3:]
    cur_s = G.aponeurosis_curves(sup, thr=0.35)
    cur_d = G.aponeurosis_curves(deep, thr=0.35)
    if not cur_s or not cur_d:
        cur = G.aponeurosis_curves(sup + deep, thr=0.35)
        if len(cur) >= 2:
            cur_s, cur_d = [cur[0]], [cur[-1]]
    if not cur_s or not cur_d or not np.isfinite(sx) or not np.isfinite(sy):
        return dict(pa_deg=fallback[0], fl_mm=fallback[1], mt_mm=fallback[2], ok=0)
    s = max(cur_s, key=lambda t: t[3]); d = max(cur_d, key=lambda t: t[3])
    mt, _, _ = G.thickness_mm([s, d], sx, sy)
    if not np.isfinite(mt):
        return dict(pa_deg=fallback[0], fl_mm=fallback[1], mt_mm=fallback[2], ok=0)
    # orientacion: se pesa cada bin por su masa entre las dos aponeurosis
    H, W = sup.shape
    yy = np.arange(H)[:, None] * np.ones((1, W))
    xx = np.ones((H, 1)) * np.arange(W)[None]
    ys = np.polyval(s[0], xx); yd = np.polyval(d[0], xx)
    dentro = (yy > np.minimum(ys, yd) + 2) & (yy < np.maximum(ys, yd) - 2)
    pesos = [float((ori[k] * dentro).sum()) for k in range(K_ORI)]
    th_px = angulo_circular(pesos)
    xref = 0.5 * (max(s[1], d[1]) + min(s[2], d[2]))
    pa, fl = G.pennation_and_length(th_px, d[0], xref, mt, sx, sy)
    return dict(pa_deg=pa, fl_mm=fl, mt_mm=float(np.clip(mt, 10.0, 50.0)), ok=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ds")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    import imutil as U
    import modelos
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"steps": ck.get("steps", 16)} if ck["arch"] in ("neuropixel", "hybrid") else {}
    model = modelos.build(ck["arch"], ck["vocab"], **kw)
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    meta = pd.read_csv(f"{a.root}/meta.csv")
    meta = meta[meta.kind == a.split]
    rows = []
    for _, r in meta.iterrows():
        f = f"{a.root}/{a.split}/img/{r['name']}.png"
        if not os.path.exists(f):
            continue
        img = U.imread_gray(f)
        p = votos(model, img, dev)
        m = medir(p, r["sx"], r["sy"])
        m["image_id"] = r["name"]
        rows.append(m)
    df = pd.DataFrame(rows)
    df["image_id"] = df.image_id.astype(str)
    df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].to_csv(a.out, index=False)
    print("escrito", a.out, len(df), "ok=%.3f" % df.ok.mean(), flush=True)
    print(df[["pa_deg", "fl_mm", "mt_mm"]].describe().round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
