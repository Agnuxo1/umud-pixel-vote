"""Relee la tabla de la ablacion ponderando cada pliegue por su numero de imagenes.

Los cinco grupos de dispositivo no son del mismo tamano, ni de lejos: 467, 139, 267, 13 y
72 imagenes con mascara de aponeurosis. La media simple de los cinco pliegues trata el
grupo de 13 imagenes igual que el de 467, y resulta que el pliegue de 13 es justo donde el
hibrido marca su mejor valor absoluto de MT (0,49 mm). Hay que comprobar dos cosas antes de
que las compruebe otro:

  1. si la ventaja del hibrido en MT sobrevive al quitar ese pliegue, y
  2. si sobrevive a ponderar cada pliegue por su n, que es la media que corresponde cuando
     lo que se promedia son errores por imagen.

Y de paso la misma lectura para el angulo, donde aparece lo contrario: la media de
NeuroPixel sola en el angulo con mascara verdadera la produce un unico pliegue.

    python ponderar_por_n.py <carpeta_de_evaluacion>

La carpeta es la que escribe `eval_all2.py` (un JSON por checkpoint). El orden de los
pliegues sale de los propios JSON, no de una lista escrita a mano.
"""
import glob
import json
import os
import statistics as st
import sys

ARCHS = ["control", "control_small", "hybrid", "neuropixel"]
EJES = [("mae_mt_mm", "n_apo", "MT (mm)"),
        ("mae_pa_libre_deg", "n_fasc", "angulo libre (deg)"),
        ("mae_pa_oracle_deg", "n_fasc", "angulo con mascara verdadera (deg)")]

carpeta = sys.argv[1] if len(sys.argv) > 1 else "runs/eval2"
D = {os.path.basename(f)[:-5]: json.load(open(f))
     for f in sorted(glob.glob(os.path.join(carpeta, "*_s*.json")))}
if not D:
    raise SystemExit(f"sin evaluaciones en {carpeta}")

semillas = sorted({int(k.rsplit("_s", 1)[1]) for k in D})
pliegues = [D[f"control_s{s}"]["val_group"] for s in semillas]
n = {eje[1]: {s: D[f"control_s{s}"][eje[1]] for s in semillas} for eje in EJES}

print("pliegue de informe y tamano:")
for s, p in zip(semillas, pliegues):
    print(f"  s{s:<4} {p:14s} n_apo={n['n_apo'][s]:4d}  n_fasc={n['n_fasc'][s]:4d}")
print(f"  {'TOTAL':20s} n_apo={sum(n['n_apo'].values()):4d}  "
      f"n_fasc={sum(n['n_fasc'].values()):4d}")

for clave, nk, titulo in EJES:
    print(f"\n--- {titulo}: media simple de pliegues vs ponderada por imagenes ---")
    tot = sum(n[nk].values())
    for a in ARCHS:
        v = [D[f"{a}_s{s}"][clave] for s in semillas]
        simple = sum(v) / len(v)
        pond = sum(x * n[nk][s] for x, s in zip(v, semillas)) / tot
        print(f"  {a:14s} simple={simple:6.2f}  ponderada={pond:6.2f}  "
              f"delta={pond - simple:+5.2f}")


def emparejado(a, b, clave, nk):
    """a - b por pliegue. Negativo = gana `a`. Es la unica comparacion valida aqui:
    cada celda tiene su propio dispositivo retenido."""
    d = [D[f"{a}_s{s}"][clave] - D[f"{b}_s{s}"][clave] for s in semillas]
    print(f"\n--- {a} - {b}, {clave} (negativo = gana {a}) ---")
    for s, p, x in zip(semillas, pliegues, d):
        print(f"  {p:14s} n={n[nk][s]:4d}  dif={x:+7.3f}")
    print(f"  mediana={st.median(d):+.3f}  media={sum(d) / len(d):+.3f}  "
          f"gana {sum(1 for x in d if x < 0)}/{len(d)}")
    grandes = [(s, x) for s, x in zip(semillas, d) if n[nk][s] >= 50]
    if len(grandes) < len(d):
        print(f"  solo pliegues con n>=50: mediana={st.median([x for _, x in grandes]):+.3f}  "
              f"gana {sum(1 for _, x in grandes if x < 0)}/{len(grandes)}")
    return d


emparejado("hybrid", "control", "mae_mt_mm", "n_apo")
emparejado("hybrid", "control", "mae_pa_oracle_deg", "n_fasc")
emparejado("neuropixel", "hybrid", "mae_pa_oracle_deg", "n_fasc")
emparejado("neuropixel", "control_small", "mae_pa_oracle_deg", "n_fasc")
