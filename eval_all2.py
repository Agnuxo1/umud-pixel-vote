"""Evalua los checkpoints de la rejilla con el arnes local y escribe una tabla.

Cada semilla entreno con un pliegue de validacion distinto (grupo de dispositivo
retenido), asi que la comparacion se hace EMPAREJADA por pliegue, no como dispersion
de semillas. Lo que se mide es error en milimetros y grados, no el IoU por clase.
"""
import glob
import json
import os
import subprocess
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
ROOT = sys.argv[2] if len(sys.argv) > 2 else "ds"
OUT = os.path.join(RUNS, "eval2")
os.makedirs(OUT, exist_ok=True)

tabla = {}
for ck in sorted(glob.glob(os.path.join(RUNS, "*.pt"))):
    tag = os.path.splitext(os.path.basename(ck))[0]
    dst = os.path.join(OUT, tag + ".json")
    if not os.path.exists(dst):
        r = subprocess.run([sys.executable, "evaluar2.py", "--root", ROOT,
                            "--ckpt", ck, "--out", dst],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("FALLO", tag, r.stderr[-800:], flush=True)
            continue
    tabla[tag] = json.load(open(dst))
    print("OK", tag, json.dumps(tabla[tag]), flush=True)

json.dump(tabla, open(os.path.join(OUT, "tabla.json"), "w"), indent=1)
# evaluar2 publica score_proxy_oracle, y lo deja en None cuando la cota de FL no es
# informativa. Antes aqui se leia "score_proxy", que no existe: la tabla final reventaba
# con KeyError justo al terminar las 20 evaluaciones.
resumen = {}
for k, v in sorted(tabla.items()):
    sp = v.get("score_proxy_oracle")
    resumen[k] = round(sp, 4) if isinstance(sp, (int, float)) else None
print("TABLA", json.dumps(resumen), flush=True)
# Lo que de verdad se compara es el error en unidades fisicas, con las dos lecturas separadas.
print("MAE", json.dumps({k: {"mt_mm": round(v["mae_mt_mm"], 3),
                             "pa_libre": round(v["mae_pa_libre_deg"], 2),
                             "pa_oracle": round(v["mae_pa_oracle_deg"], 2),
                             "pliegue": v["val_group"]}
                         for k, v in sorted(tabla.items())}, indent=1), flush=True)
