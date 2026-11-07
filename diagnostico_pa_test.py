"""Por que el pipeline propio puntua 4,45 en el test: PA sale bajo y FL satura el techo.

El envio del 31-10 dio **4,44858** frente a 0,23862 de un promedio de notebooks publicos.
Descomponiendo el score con el ensemble como referencia aproximada, FL aporta 9,41 de los
13,35 (4,45 x 3), MT 3,13 y PA 1,29: el culpable es FL. Y el CSV enviado tiene **fl_mm con
mediana 200,0**, que es exactamente el techo del recorte, porque FL = MT / sin(PA) explota
cuando PA es pequeno. PA enviado: mediana 8,17 grados; el ensemble: 15,92.

Hipotesis: `infer.medir` promedia la orientacion sobre **toda** la region entre aponeurosis
(`ori[k] * dentro`). La mayoria de esos pixeles no son fasciculo, y su masa de orientacion
arrastra el angulo hacia el bin horizontal, encogiendo PA hacia 0. La lectura que el arnes
interno valido es la que pesa por la masa de fasciculo predicha. Si la hipotesis es correcta,
al pesar por esa masa PA sube y FL deja de saturar.

    python diagnostico_pa_test.py --ckpt <checkpoint.pt> [--max N]

La lectura A llama a `infer.medir` sin tocarla, para reproducir el envio exactamente; la B
repite su geometria cambiando solo los pesos del promedio circular. No se ajusta nada contra el
leaderboard: son dos lecturas de la misma salida del mismo modelo.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import geometria as G
import imutil as U
import infer as I
import modelos
from train import K_ORI


def medir_pesando_fasciculo(p, sx, sy, fallback=(15.0, 70.0, 20.0)):
    """Igual que `infer.medir` pero pesando cada bin por la masa de fasciculo del pixel.

    La unica linea que cambia es la de `pesos`. Todo lo demas se copia de `infer.medir` para
    que la comparacion aisle la lectura y nada mas.
    """
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
    s = max(cur_s, key=lambda t: t[3])
    d = max(cur_d, key=lambda t: t[3])
    mt, _, _ = G.thickness_mm([s, d], sx, sy)
    if not np.isfinite(mt):
        return dict(pa_deg=fallback[0], fl_mm=fallback[1], mt_mm=fallback[2], ok=0)
    H, W = sup.shape
    yy = np.arange(H)[:, None] * np.ones((1, W))
    xx = np.ones((H, 1)) * np.arange(W)[None]
    ys = np.polyval(s[0], xx)
    yd = np.polyval(d[0], xx)
    dentro = (yy > np.minimum(ys, yd) + 2) & (yy < np.maximum(ys, yd) - 2)
    masa = ori.sum(0)                                   # <-- la unica diferencia
    pesos = [float((ori[k] * dentro * masa).sum()) for k in range(K_ORI)]
    th_px = I.angulo_circular(pesos)
    xref = 0.5 * (max(s[1], d[1]) + min(s[2], d[2]))
    pa, fl = G.pennation_and_length(th_px, d[0], xref, mt, sx, sy)
    return dict(pa_deg=pa, fl_mm=fl, mt_mm=float(np.clip(mt, 10.0, 50.0)), ok=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="ds")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default="diagnostico_pa_test.csv")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    dev = os.environ.get("UMUD_DEV", "cpu")   # la GPU local solo con autorizacion de Fran
    kw = {"steps": ck.get("steps", 16)} if ck["arch"] in ("neuropixel", "hybrid") else {}
    model = modelos.build(ck["arch"], ck["vocab"], **kw)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    print(f"arch={ck['arch']} val_group={ck.get('val_group')} dev={dev}", flush=True)

    meta = pd.read_csv(f"{a.root}/meta.csv")
    meta = meta[meta.kind == a.split]
    if a.max:
        meta = meta.head(a.max)
    print(f"{a.split}: {len(meta)} imagenes", flush=True)

    filas = []
    for i, r in enumerate(meta.itertuples()):
        f = f"{a.root}/{a.split}/img/{r.name}.png"
        if not os.path.exists(f):
            continue
        img = U.imread_gray(f)
        p = I.votos(model, img, dev)
        A = I.medir(p, r.sx, r.sy)                  # la del envio, sin tocar
        B = medir_pesando_fasciculo(p, r.sx, r.sy)  # la que valido el arnes
        filas.append({"image_id": r.name,
                      "pa_region": A["pa_deg"], "fl_region": A["fl_mm"], "mt_region": A["mt_mm"], "ok_region": A["ok"],
                      "pa_fasc": B["pa_deg"], "fl_fasc": B["fl_mm"], "mt_fasc": B["mt_mm"], "ok_fasc": B["ok"]})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(meta)}", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(a.out, index=False)
    print(f"\n{len(df)} filas -> {a.out}")
    print(f"geometria resuelta: region {int(df.ok_region.sum())}, fasciculo {int(df.ok_fasc.sum())} de {len(df)}")
    for c in ("pa_region", "pa_fasc", "fl_region", "fl_fasc", "mt_region", "mt_fasc"):
        v = df[c].dropna()
        sat = int((v >= 199.99).sum()) if c.startswith("fl") else 0
        print(f"  {c:11s} med={v.median():7.2f}  p05={v.quantile(.05):7.2f}  p95={v.quantile(.95):7.2f}"
              + (f"   saturadas en 200: {sat}/{len(v)}" if c.startswith("fl") else ""))
    print("\nreferencia  envio propio (4,44858): pa med 8,17  fl med 200,00  mt med 30,22")
    print("referencia  ensemble  (0,23862): pa med 15,92  fl med  87,09  mt med 20,83")
    print("rangos oficiales del test:       PA 5-45  FL 30-200  MT 10-50")


if __name__ == "__main__":
    main()
