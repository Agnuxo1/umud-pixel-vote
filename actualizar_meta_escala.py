"""Actualiza SOLO las columnas de escala de ds/meta.csv con una pasada nueva de scale_all.

Por que no se vuelve a ejecutar prep_dataset.py: ese reescribe los 4118 PNG de 384x256, y
las imagenes no han cambiado — solo la escala fisica, que vive en unas pocas columnas. Aqui
se recalculan exactamente con la misma formula que usa prep_dataset (`sx * W / Wi`), se
empareja por (kind, name) y se COMPRUEBA que ninguna otra columna se ha movido y que todas
las filas encontraron pareja. Si algo no cuadra, no escribe nada.

Uso: python actualizar_meta_escala.py [csv_de_escala]
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

UMUD_WORK = os.environ.get("UMUD_WORK", "E:/UMUD-work")
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{UMUD_WORK}/scale_all.csv"
META = f"{UMUD_WORK}/ds/meta.csv"
W, H = 384, 256                     # la rejilla de trabajo de prep_dataset.py

COLS_ESCALA = ["sx", "sy", "sx_img", "sy_img", "tick_y", "cov_y", "band_y", "span_y"]


def main():
    meta = pd.read_csv(META)
    esc = pd.read_csv(CSV).set_index(["kind", "name"])
    falta = [k for k in zip(meta.kind, meta.name) if k not in esc.index]
    if falta:
        print(f"ABORTA: {len(falta)} filas de meta.csv sin pareja en la escala, p.ej. {falta[:3]}")
        return 1

    nuevo = meta.copy()
    r = esc.loc[list(zip(meta.kind, meta.name))]
    # misma formula que prep_dataset: px/mm de la imagen distribuida a la rejilla de trabajo
    nuevo["sx_img"] = r["sx"].to_numpy()
    nuevo["sy_img"] = r["sy"].to_numpy()
    nuevo["sx"] = np.where(np.isfinite(nuevo.sx_img), nuevo.sx_img * W / meta.Wi, np.nan)
    nuevo["sy"] = np.where(np.isfinite(nuevo.sy_img), nuevo.sy_img * H / meta.Hi, np.nan)
    for c in ("tick_y", "cov_y", "band_y", "span_y"):
        if c in r.columns:
            nuevo[c] = r[c].to_numpy()

    intactas = [c for c in meta.columns if c not in COLS_ESCALA]
    for c in intactas:
        a, b = meta[c], nuevo[c]
        if not a.equals(b):
            print(f"ABORTA: la columna {c} no deberia cambiar y ha cambiado")
            return 1
    print(f"columnas intactas verificadas: {len(intactas)} ({', '.join(intactas)})")

    cambia = (~np.isclose(meta.sx.fillna(-1), nuevo.sx.fillna(-1), rtol=1e-6)).sum()
    print(f"filas con sx distinto: {cambia} de {len(meta)}")
    shutil.copy(META, META + ".antes_de_la_escala_v2")
    nuevo.to_csv(META, index=False)
    print(f"escrito {META} (respaldo en {os.path.basename(META)}.antes_de_la_escala_v2)")
    print(nuevo.groupby("kind").agg(n=("name", "size"),
                                    escala=("sy", lambda s: round(float(s.notna().mean()), 3))).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
