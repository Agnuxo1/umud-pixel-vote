"""Compara el margen de la ablacion con el ruido de inicializacion, dentro de cada pliegue.

La pasada publicada tiene una semilla por pliegue, porque `train.py` deriva el pliegue de la
semilla. `run_semillas.py` anade semillas dentro de un pliegue fijo. Aqui se juntan las dos
cosas y se responde la pregunta que el informe dejaba abierta: la diferencia hibrido - control
es mayor o menor que la variacion entre semillas de la misma celda?

    python dispersion_semilla.py <eval_original> <pliegue:carpeta> [<pliegue:carpeta> ...]
    python dispersion_semilla.py E:/UMUD-work/eval237 1152.0x864.0:E:/UMUD-work/sem_f0/eval2

El pliegue original se identifica por su `val_group`, no por el numero de semilla, para que no
importe el orden de `usables`.
"""
import glob
import json
import os
import statistics as st
import sys

ARCHS = ["control", "control_small", "hybrid", "neuropixel"]
EJES = [("mae_mt_mm", "MT (mm)"), ("mae_pa_libre_deg", "angulo libre (deg)"),
        ("mae_pa_oracle_deg", "angulo con mascara verdadera (deg)")]

if len(sys.argv) < 3:
    raise SystemExit(__doc__)

orig_dir = sys.argv[1]
orig = {}
for f in glob.glob(os.path.join(orig_dir, "*_s*.json")):
    d = json.load(open(f))
    tag = os.path.basename(f)[:-5]
    orig[(d.get("val_group"), tag.rsplit("_s", 1)[0])] = d

celdas = {}          # (pliegue, arch) -> [por semilla]
for spec in sys.argv[2:]:
    pliegue, carpeta = spec.split(":", 1)
    for f in glob.glob(os.path.join(carpeta, "*_s*.json")):
        d = json.load(open(f))
        if str(d.get("val_group")) != pliegue:
            raise SystemExit(f"{f}: val_group={d.get('val_group')} pero se declaro {pliegue}")
        arch = os.path.basename(f)[:-5].rsplit("_s", 1)[0]
        celdas.setdefault((pliegue, arch), []).append(d)
    for arch in ARCHS:
        o = orig.get((pliegue, arch))
        if o is None:
            print(f"AVISO: falta la celda original de {arch} en {pliegue}")
        else:
            celdas.setdefault((pliegue, arch), []).append(o)

pliegues = sorted({p for p, _ in celdas})
print(f"pliegues con dispersion medida: {len(pliegues)} -> {pliegues}")
print(f"semillas por celda: {sorted({len(v) for v in celdas.values()})}")

for clave, titulo in EJES:
    print(f"\n==================== {titulo} ====================")
    print(f"{'pliegue':15s} {'arquitectura':15s} {'n':>2} {'media':>8} {'desv':>8} {'rango':>8}   valores")
    for p in pliegues:
        for arch in ARCHS:
            v = [c[clave] for c in celdas.get((p, arch), []) if isinstance(c.get(clave), (int, float))]
            if not v:
                continue
            desv = st.stdev(v) if len(v) > 1 else float("nan")
            print(f"{p:15s} {arch:15s} {len(v):2d} {sum(v)/len(v):8.3f} {desv:8.3f} "
                  f"{max(v)-min(v):8.3f}   {[round(x, 2) for x in v]}")

    print(f"\n  -- hibrido - control, frente a la dispersion de semilla --")
    for p in pliegues:
        h = [c[clave] for c in celdas.get((p, "hybrid"), []) if isinstance(c.get(clave), (int, float))]
        c0 = [c[clave] for c in celdas.get((p, "control"), []) if isinstance(c.get(clave), (int, float))]
        if len(h) < 2 or len(c0) < 2:
            continue
        dif = sum(h)/len(h) - sum(c0)/len(c0)
        # Desviacion tipica agrupada de las dos celdas, y error tipico de la diferencia de medias.
        sp = (((len(h)-1)*st.variance(h) + (len(c0)-1)*st.variance(c0)) / (len(h)+len(c0)-2)) ** 0.5
        et = sp * (1/len(h) + 1/len(c0)) ** 0.5
        veredicto = "SUPERA el ruido" if abs(dif) > 2*et else ("dentro del ruido" if abs(dif) < et else "del orden del ruido")
        print(f"  {p:15s} dif={dif:+7.3f}  desv_agrupada={sp:6.3f}  error_tipico={et:6.3f}  "
              f"|dif|/et={abs(dif)/et if et else float('inf'):5.2f}  -> {veredicto}")
