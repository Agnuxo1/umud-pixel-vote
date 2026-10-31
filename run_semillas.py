"""Dispersion de semilla dentro de un pliegue fijo, y evaluacion en el mismo trabajo.

`run_all.py` recorre semillas, pero el pliegue de informe sale de la semilla
(`vg = usables[seed % 5]` en `train.py`), asi que sus cinco "semillas" son en realidad
los cinco pliegues de dispositivo retenido: una semilla por pliegue y ninguna medida de
la dispersion de inicializacion. Eso deja sin responder la pregunta obvia sobre la
ablacion: la mediana de -0,26 mm en MT entre el hibrido y el control, esta por encima
del ruido de semilla o por debajo?

Este guion fija el pliegue con `--val-group` y mueve solo la semilla. El grupo de
seleccion se deriva de forma determinista del de informe dentro de `train.py`, de modo
que las celdas nuevas son comparables con las de la pasada original sin tocar su codigo.

    python -u run_semillas.py <val_group> [archs] [seeds] [epochs]
    python -u run_semillas.py 1152.0x864.0 control,control_small,neuropixel,hybrid 100,200 12

Las semillas van en los cientos para que ningun `arch_sN` nuevo se confunda con los de
la pasada original, que son `s0`..`s4`. Al final evalua los checkpoints del pliegue en
milimetros y grados, para que el trabajo devuelva la medida y no haya que descargar los
pesos.
"""
import itertools
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    raise SystemExit(__doc__)

VAL_GROUP = sys.argv[1]
ARCHS = (sys.argv[2] if len(sys.argv) > 2 else "control,control_small,neuropixel,hybrid").split(",")
SEEDS = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "100,200").split(",")]
EPOCHS = sys.argv[4] if len(sys.argv) > 4 else "12"

OUT = os.path.join(ROOT, "runs_sem")
os.makedirs(OUT, exist_ok=True)
print(f"pliegue={VAL_GROUP} archs={ARCHS} semillas={SEEDS} epocas={EPOCHS}", flush=True)

for arch, seed in itertools.product(ARCHS, SEEDS):
    tag = f"{arch}_s{seed}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        print("ya hecho", tag, flush=True)
        continue
    cmd = [sys.executable, "-u", os.path.join(ROOT, "train.py"),
           "--root", os.path.join(ROOT, "ds"), "--arch", arch, "--seed", str(seed),
           "--epochs", EPOCHS, "--val-group", VAL_GROUP, "--out", OUT]
    # Prueba de humo antes de encolar: recorta cada epoca a N lotes para comprobar en
    # local que el pliegue llega y que la verificacion de abajo muerde.
    if os.environ.get("UMUD_LIMITE_LOTES"):
        cmd += ["--limite-lotes", os.environ["UMUD_LIMITE_LOTES"]]
    print("====", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("FALLO", tag, r.returncode, flush=True)

hechos = sorted(f[:-5] for f in os.listdir(OUT) if f.endswith(".json") and f != "resumen.json")
print("ENTRENADOS", json.dumps(hechos), flush=True)

# Comprobacion dura: si train.py hubiera ignorado --val-group, la dispersion medida seria
# entre pliegues distintos y no diria nada. Mejor que reviente aqui que publicar eso.
for tag in hechos:
    d = json.load(open(os.path.join(OUT, tag + ".json")))
    if str(d.get("val_group")) != VAL_GROUP:
        raise SystemExit(f"{tag}: val_group={d.get('val_group')} y se pidio {VAL_GROUP}")
print("VERIFICADO: todas las celdas comparten el pliegue de informe", VAL_GROUP, flush=True)

r = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "eval_all2.py"), OUT,
                    os.path.join(ROOT, "ds")], cwd=ROOT)
print("EVAL exit", r.returncode, flush=True)
