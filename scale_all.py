"""Una sola pasada de escala sobre TODAS las imagenes (entrenamiento y test), en paralelo."""
import glob, os
from multiprocessing import Pool

import pandas as pd

import os

UMUD_RAW = os.environ.get("UMUD_RAW", "E:/UMUD-data/raw")
UMUD_WORK = os.environ.get("UMUD_WORK", "E:/UMUD-work")

RAW = UMUD_RAW
SETS = {
    "apo":  f"{RAW}/apo_imgs_v1/apo_images_new_model_v1/*",
    "fasc": f"{RAW}/fasc_imgs_v1/fasc_images_new_model_v1/*",
    "test": f"{RAW}/test_images_v2/test_set_v2/*",
}
OUT = f"{UMUD_WORK}/scale_all.csv"


def work(args):
    import roi_scale as S
    kind, f = args
    r = S.measure(f)
    r["kind"] = kind
    r["name"] = os.path.splitext(os.path.basename(f))[0]
    return r


if __name__ == "__main__":
    tasks = [(k, f) for k, pat in SETS.items()
             for f in sorted(glob.glob(pat)) if f.lower().endswith((".tif", ".png"))]
    print("imagenes", len(tasks), flush=True)
    rows = []
    # El numero de procesos es configurable: con 6 y ~4 GB libres el PC se queda sin
    # memoria a mitad de pasada (ArrayMemoryError al reservar 800x652 float32).
    nproc = int(os.environ.get("UMUD_PROCS", "3"))
    with Pool(nproc) as p:
        for i, r in enumerate(p.imap_unordered(work, tasks, chunksize=8)):
            rows.append(r)
            if i % 400 == 0:
                print(i, len(tasks), flush=True)
    df = pd.DataFrame(rows).sort_values(["kind", "name"])
    df.to_csv(OUT, index=False)
    print("TOTAL", len(df), flush=True)
    print(df.groupby("kind").agg(n=("sy", "size"), sy_ok=("sy", lambda s: float(s.notna().mean()))).round(3).to_string(), flush=True)
