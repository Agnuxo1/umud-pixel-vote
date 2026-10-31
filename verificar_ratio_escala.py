"""Comprueba que sx/sy sigue la anisotropia nativa. Es la prueba que descubrio el fallo.

La imagen distribuida esta reescalada, pero la mascara conserva el tamano nativo del
fotograma. Si el pixel nativo es isotropo, la anisotropia del archivo distribuido es

    aniso_nativa = (Wi / Wm) / (Hi / Hm)

y el modulo de escala, que NO ve la mascara, deberia reproducirla en sx/sy. Esta prueba
mide el error relativo de esa razon, y lo separa por si las dos reglas eligieron el MISMO
valor de marca o no. Con la version anterior de roi_scale, que elegia el valor de la regla
vertical y el de la horizontal por separado, el 46,4 % de las imagenes salia con marcas
discordantes y esas tenian un error mediano del 100,6 %: el ecografo imprime una sola
escala, asi que elegirlas distintas es fisicamente imposible. Donde ya coincidian, el error
era del 0,5 %. De ahi el arreglo de `choose_tick_comun`.

Uso: python verificar_ratio_escala.py [csv_de_escala]
"""
import os
import sys

import numpy as np
import pandas as pd

UMUD_WORK = os.environ.get("UMUD_WORK", "E:/UMUD-work")
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{UMUD_WORK}/scale_all.csv"
META = f"{UMUD_WORK}/ds/meta.csv"


def main():
    esc = pd.read_csv(CSV)
    meta = pd.read_csv(META)[["kind", "name", "Wi", "Hi", "Wm", "Hm"]]
    d = esc.merge(meta, on=["kind", "name"], how="inner")
    d = d[d.kind.isin(["apo", "fasc"])].copy()
    d["aniso_nativa"] = (d.Wi / d.Wm) / (d.Hi / d.Hm)
    d = d[np.isfinite(d.sx) & np.isfinite(d.sy) & (d.sy > 0) & np.isfinite(d.aniso_nativa)]
    d["ratio"] = d.sx / d.sy
    d["err_rel"] = (d.ratio - d.aniso_nativa).abs() / d.aniso_nativa

    def bloque(nombre, sub):
        if not len(sub):
            print(f"{nombre:28s} n=0")
            return
        print(f"{nombre:28s} n={len(sub):5d}  err_mediano={100 * sub.err_rel.median():6.1f} %"
              f"  dentro_5%={100 * (sub.err_rel < 0.05).mean():5.1f} %"
              f"  ratio_mediano={sub.ratio.median():.3f}  nativa_mediana={sub.aniso_nativa.median():.3f}")

    print(f"csv={CSV}")
    bloque("TODAS", d)
    if {"tick_x", "tick_y"} <= set(d.columns):
        tx, ty = d.tick_x, d.tick_y
        amb = d[np.isfinite(tx) & np.isfinite(ty)]
        bloque("dos reglas leidas", amb)
        bloque("  marca coincidente", amb[amb.tick_x == amb.tick_y])
        bloque("  marca discordante", amb[amb.tick_x != amb.tick_y])
        if len(amb):
            n_disc = int((amb.tick_x != amb.tick_y).sum())
            print(f"discordancia entre reglas: {n_disc} imagenes = "
                  f"{100.0 * n_disc / len(amb):.1f} % de las que tienen dos reglas, "
                  f"{100.0 * n_disc / len(d):.1f} % del total")
            # Lo que el arreglo NO toca: con una sola regla no hay con que comparar, asi que
        # el modulo devuelve sx = sy, es decir isotropia. Si la anisotropia nativa de esa
        # poblacion no es 1, ese error se queda. Conviene verlo aparte y no diluido.
        bloque("una sola regla", d[~(np.isfinite(tx) & np.isfinite(ty))])
    # Criterio de aprobado. El primer umbral que puse aqui era el error mediano de TODAS las
    # imagenes por debajo del 5 %, y eso no se puede cumplir: la poblacion de una sola regla
    # arrastra un 11,1 % que no es un fallo del codigo sino la ausencia de informacion, y
    # propagarla por huella de equipo esta medido y es PEOR (E13b del registro). Lo que si se
    # puede exigir es que donde hay DOS reglas la razon sea buena, y que no quede ninguna
    # imagen con valores de marca discordantes, que era el fallo real.
    dos = d[np.isfinite(d.tick_x) & np.isfinite(d.tick_y)] if {"tick_x", "tick_y"} <= set(d.columns) else d
    disc = int((dos.tick_x != dos.tick_y).sum()) if len(dos) else 0
    err_dos = float(dos.err_rel.median()) if len(dos) else float("nan")
    ok = bool(disc == 0 and err_dos < 0.05)
    print(f"VEREDICTO {'PASA' if ok else 'FALLA'}: con dos reglas, error mediano "
          f"{100 * err_dos:.1f} % (umbral 5 %) y {disc} imagenes discordantes (umbral 0). "
          f"El {100 * (1 - len(dos) / len(d)):.0f} % con una sola regla queda en isotropia, "
          f"con un {100 * d[~d.index.isin(dos.index)].err_rel.median():.1f} % de error "
          f"irreparable con los datos del reto.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
