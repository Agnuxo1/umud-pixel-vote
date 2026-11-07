"""Genera el envio sin propagar la saturacion de FL cuando PA cae en el recorte.

Diagnostico (E20): en el test el cabezal de orientacion predice **horizontal con alta
confianza** — concentracion circular R = 0,99 y angulo mediano 0,4 grados — en dispositivos que
no ha visto. Eso no es falta de señal ni un error de lectura ni de escala: es una falla de
generalizacion dirigida, coherente con que el cabezal sea dependiente del dispositivo (gana en
512x512 y no en 1152x864, medido con 3 semillas).

El problema de ingenieria es lo que la geometria hace con ese fallo. `PA = |a_f - a_d|` sale
~1 grado, el recorte lo sube a `pa_lo = 5`, y entonces `FL = MT / sin(PA)` da 200 mm, el techo.
En 18 de 30 imagenes del test FL sale saturado. Publicar el techo de un recorte no es una
prediccion: es un artefacto que multiplica el error (|200 - 87| / 12 = 9,4 de los 13,35 puntos
del score observado).

La regla de aqui: **si PA toca el recorte inferior, el estimador declara que no tiene medida de
PA**, y se usa el respaldo fisiologico que ya existe en `infer.medir` en lugar de dividir por
sin(PA). No es un hiperparametro ni esta ajustado contra el leaderboard: "PA toco el recorte" es
una condicion binaria de la propia formula, y el respaldo (15 grados, 70 mm) ya estaba en el
codigo para el caso de geometria no resuelta. MT no se toca: es medida directa de las dos
curvas y no depende de PA.

    python envio_sin_saturar.py --ckpt <checkpoint.pt> [--root ds] [--out envio.csv]

Escribe tambien `<out>.comparacion.csv` con las dos variantes por imagen, para que la diferencia
sea auditable.
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

PA_LO, PA_HI = 5.0, 45.0
FALLBACK = (15.0, 70.0, 20.0)


def medir_sin_saturar(p, sx, sy):
    """`infer.medir`, pero sin propagar FL cuando PA cae en el recorte inferior."""
    base = I.medir(p, sx, sy, fallback=FALLBACK)
    if not base["ok"]:
        return base, "geometria_no_resuelta"
    if base["pa_deg"] <= PA_LO + 1e-6:
        # El angulo del fasciculo no se distingue del de la aponeurosis: no hay medida de PA,
        # asi que FL = MT / sin(PA) no es aplicable. Se declara con el respaldo fisiologico.
        return dict(pa_deg=FALLBACK[0], fl_mm=FALLBACK[1], mt_mm=base["mt_mm"], ok=1), "pa_en_recorte"
    if base["fl_mm"] >= 199.99 or base["fl_mm"] <= 30.01:
        # FL tambien saturado por el otro lado: misma razon, el divisor no es de fiar.
        return dict(pa_deg=base["pa_deg"], fl_mm=FALLBACK[1], mt_mm=base["mt_mm"], ok=1), "fl_en_recorte"
    return base, "medido"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="ds")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="envio_sin_saturar.csv")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    dev = os.environ.get("UMUD_DEV", "cpu")
    kw = {"steps": ck.get("steps", 16)} if ck["arch"] in ("neuropixel", "hybrid") else {}
    model = modelos.build(ck["arch"], ck["vocab"], **kw)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    print(f"arch={ck['arch']} val_group={ck.get('val_group')} dev={dev}", flush=True)

    meta = pd.read_csv(f"{a.root}/meta.csv")
    meta = meta[meta.kind == a.split]
    print(f"{a.split}: {len(meta)} imagenes", flush=True)

    filas, comp = [], []
    for i, r in enumerate(meta.itertuples()):
        f = f"{a.root}/{a.split}/img/{r.name}.png"
        if not os.path.exists(f):
            continue
        p = I.votos(model, U.imread_gray(f), dev)
        A = I.medir(p, r.sx, r.sy, fallback=FALLBACK)
        B, motivo = medir_sin_saturar(p, r.sx, r.sy)
        filas.append({"image_id": r.name, "pa_deg": round(B["pa_deg"], 4),
                      "fl_mm": round(B["fl_mm"], 4), "mt_mm": round(B["mt_mm"], 4)})
        comp.append({"image_id": r.name, "motivo": motivo,
                     "pa_antes": A["pa_deg"], "fl_antes": A["fl_mm"], "mt_antes": A["mt_mm"],
                     "pa_despues": B["pa_deg"], "fl_despues": B["fl_mm"], "mt_despues": B["mt_mm"]})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(meta)}", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(a.out, index=False)
    dc = pd.DataFrame(comp)
    dc.to_csv(a.out + ".comparacion.csv", index=False)
    print(f"\n{len(df)} filas -> {a.out}")
    print("motivos:", dc.motivo.value_counts().to_dict())
    for c in ("pa_deg", "fl_mm", "mt_mm"):
        print(f"  {c:7s} med={df[c].median():7.2f}  p05={df[c].quantile(.05):7.2f}  p95={df[c].quantile(.95):7.2f}")
    sat = int((dc.fl_antes >= 199.99).sum())
    print(f"\nFL saturado en 200 antes: {sat}/{len(dc)}  ->  despues: {int((df.fl_mm >= 199.99).sum())}")
    print("referencia  envio propio (4,44858): pa 8,17  fl 200,00  mt 30,22")
    print("referencia  ensemble  (0,23862): pa 15,92  fl  87,09  mt 20,83")


if __name__ == "__main__":
    main()
