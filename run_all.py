"""Lanza el experimento completo en el carril: control, neuropixel e hibrido x 3 semillas.

Un solo trabajo de cola para no reenviar el conjunto de datos nueve veces. Cada combinacion
escribe su propio checkpoint y su propio JSON, asi que el trabajo se puede reanudar.
"""
import itertools
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# El directorio de salida es un argumento para que la pasada corregida no pise los
# checkpoints viejos: aquellos se eligieron en el pliegue informado y no sirven, pero
# se conservan como registro de lo que la auditoria encontro.
OUT = os.path.join(ROOT, os.environ.get("UMUD_OUT", "runs"))
os.makedirs(OUT, exist_ok=True)

ARCHS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["control", "neuropixel", "control_small", "hybrid"]
SEEDS = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1", "2", "3", "4"])]
EPOCHS = sys.argv[3] if len(sys.argv) > 3 else "10"

for arch, seed in itertools.product(ARCHS, SEEDS):
    tag = f"{arch}_s{seed}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        print("ya hecho", tag, flush=True)
        continue
    cmd = [sys.executable, "-u", os.path.join(ROOT, "train.py"), "--root", os.path.join(ROOT, "ds"),
           "--arch", arch, "--seed", str(seed), "--epochs", EPOCHS, "--out", OUT]
    print("====", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("FALLO", tag, r.returncode, flush=True)

res = {}
for f in sorted(os.listdir(OUT)):
    if f.endswith(".json"):
        res[f[:-5]] = json.load(open(os.path.join(OUT, f)))
json.dump(res, open(os.path.join(OUT, "resumen.json"), "w"), indent=1)
print("RESUMEN", json.dumps({k: round(v.get("best_sel_media3", v.get("best_val", float("nan"))), 4)
                             for k, v in res.items()}), flush=True)
