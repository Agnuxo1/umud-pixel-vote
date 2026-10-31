"""Valida la anisotropia por huella dejando fuera el dispositivo entero.

La prueba facil seria estimar la mediana de cada huella con todas las filas y medir el error
en esas mismas filas: eso se aprueba solo. La prueba que corresponde al problema es dejar
fuera un GRUPO de fotograma nativo completo, aprender sin el y medir en el, porque es
exactamente lo que pasa con una imagen de test de un equipo no visto. Se informan las dos,
y la que vale es la segunda.
"""
import os
import sys

import numpy as np
import pandas as pd

import aniso_nativa as A

UMUD_WORK = os.environ.get("UMUD_WORK", "E:/UMUD-work")
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{UMUD_WORK}/scale_all.csv"


def carga():
    esc = pd.read_csv(CSV)
    meta = pd.read_csv(f"{UMUD_WORK}/ds/meta.csv")[["kind", "name", "Wm", "Hm"]]
    return esc.merge(meta, on=["kind", "name"], how="left")


def err(d):
    v = d[d.kind.isin(["apo", "fasc"])].copy()
    v["real"] = (v.W / v.Wm) / (v.H / v.Hm)
    v = v[np.isfinite(v.real) & np.isfinite(v.sx) & np.isfinite(v.sy) & (v.sy > 0)]
    e = ((v.sx / v.sy) - v.real).abs() / v.real
    return v, e


def linea(nombre, v, e, n_total):
    print(f"{nombre:34s} n={len(v):5d}/{n_total}  err_mediano={100 * e.median():6.2f} %"
          f"  dentro_5%={100 * (e < 0.05).mean():5.1f} %  p90={100 * e.quantile(.9):6.1f} %")


def main():
    d = carga()
    base, e0 = err(d)
    n_tot = len(d[d.kind.isin(["apo", "fasc"])])
    print(f"csv={CSV}\n")
    linea("CRUDA (como sale de la pasada)", base, e0, n_tot)

    opt = A.aplica(d, A.aprende(d))
    v1, e1 = err(opt)
    linea("HUELLA, optimista (se aprueba sola)", v1, e1, n_tot)

    # honesta: cada grupo de fotograma nativo se mide con tablas que no lo han visto
    grupos = sorted(set(zip(d.Wm.dropna(), d.Hm.dropna())))
    trozos = []
    for g in grupos:
        tablas = A.aprende(d, excluir_grupos=[g])
        sub = d[(d.Wm == g[0]) & (d.Hm == g[1])]
        if len(sub):
            trozos.append(A.aplica(sub, tablas))
    hon = pd.concat(trozos, ignore_index=True)
    v2, e2 = err(hon)
    linea("HUELLA, dejando fuera el equipo", v2, e2, n_tot)
    print(f"\n{len(grupos)} grupos de fotograma nativo; cada uno medido con tablas que no lo vieron")
    print("origen de sx (version honesta):", hon.sx_src.value_counts(dropna=False).to_dict())
    print("origen de sy (version honesta):", hon.sy_src.value_counts(dropna=False).to_dict())

    te = opt[opt.kind == "test"]
    print(f"\ncobertura de sy en test: cruda {100 * d[d.kind == 'test'].sy.notna().mean():.1f} %"
          f" -> con huella {100 * te.sy.notna().mean():.1f} %")
    print("origen de sx en test:", te.sx_src.value_counts(dropna=False).to_dict())
    mejor = e2.median() < e0.median() - 0.005
    print(f"\nVEREDICTO {'USAR' if mejor else 'NO USAR'}: el error que cuenta es el de dejar "
          f"fuera el equipo, {100 * e0.median():.2f} % -> {100 * e2.median():.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
