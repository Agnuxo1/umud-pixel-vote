"""Comprobacion de la geometria con un caso construido cuya respuesta se conoce.

Dos aponeurosis horizontales separadas 60 px con 6 px/mm -> MT = 10 mm exactos.
Fasciculos a 20 grados sobre una aponeurosis horizontal -> PA = 20 grados y
FL = MT / sin(20) = 29,24 mm. Si la tuberia no reproduce esto, no mide lo que dice medir.
"""
import numpy as np

import geometria as G
import infer as I
from train import K_ORI, VOCAB


def campo(H=256, W=384, y_sup=60, y_deep=180, ang=20.0, sx=6.0, sy=6.0):
    p = np.zeros((VOCAB, H, W), np.float32)
    p[0] = 1.0
    for y, c in ((y_sup, 1), (y_deep, 2)):
        p[c, y - 1:y + 2, 40:W - 40] = 1.0
        p[0, y - 1:y + 2, 40:W - 40] = 0.0
    # bin de orientacion correspondiente a -ang (fasciculo ascendente hacia la derecha)
    b = int(np.floor((ang + 90.0) / (180.0 / K_ORI)))
    p[3 + b, y_sup + 3:y_deep - 3, 50:W - 50] = 1.0
    p[0, y_sup + 3:y_deep - 3, 50:W - 50] = 0.0
    return p


def test_todo():
    sx = sy = 6.0
    ang = 20.0
    p = campo(sx=sx, sy=sy, ang=ang)
    r = I.medir(p, sx, sy)
    mt_esperado = (180 - 60) / sy
    assert r["ok"] == 1, r
    assert abs(r["mt_mm"] - mt_esperado) < 0.6, (r["mt_mm"], mt_esperado)
    assert abs(r["pa_deg"] - ang) <= 180.0 / K_ORI, (r["pa_deg"], ang)   # +-un bin
    fl = mt_esperado / np.sin(np.radians(r["pa_deg"]))
    assert abs(r["fl_mm"] - fl) < 1.0, (r["fl_mm"], fl)
    print("geometria OK:", {k: round(v, 2) for k, v in r.items()})

    # el pixel anisotropo cambia el angulo: con sx=2*sy el fasciculo se ve mas tumbado
    a_iso = G.fascicle_angle_deg(20.0, 6.0, 6.0)
    a_ani = G.fascicle_angle_deg(20.0, 12.0, 6.0)
    assert abs(a_iso - 20.0) < 1e-6, a_iso
    assert a_ani > a_iso + 10, (a_iso, a_ani)
    print("anisotropia OK:", round(a_iso, 2), round(a_ani, 2))

    # FL crece cuando PA baja
    _, fl1 = G.pennation_and_length(10.0, np.array([0.0, 0.0, 100.0]), 100, 20.0, 6.0, 6.0)
    _, fl2 = G.pennation_and_length(30.0, np.array([0.0, 0.0, 100.0]), 100, 20.0, 6.0, 6.0)
    assert fl1 > fl2, (fl1, fl2)
    print("FL OK:", round(fl1, 1), round(fl2, 1))


def test_orientacion_linea_fina():
    """Regresion: la orientacion de una linea FINA, que es lo que son las mascaras reales.

    Las mascaras de fasciculo tienen 2 px de grosor medio. En una linea asi el gradiente de
    Sobel es ~0 sobre la propia linea y toda la energia esta en los vecinos. Una version
    anterior pesaba el tensor por la mascara y borraba esa energia justo donde estaba: una
    linea horizontal devolvia -90 grados, el error maximo posible, y el error medio era de
    54,7 grados. El fallo no lo cazo ningun test porque los casos construidos usaban trazos
    gruesos. Este test usa lineas de 1 px, que es el regimen real.
    """
    errs = []
    for real in (-60.0, -45.0, 0.0, 45.0, 60.0):
        H = W = 128
        ys = (H / 2 + np.tan(np.radians(real)) * (np.arange(W) - W / 2)).round().astype(int)
        f = np.zeros((H, W), np.float32)
        ok = (ys >= 0) & (ys < H)
        f[ys[ok], np.arange(W)[ok]] = 1.0
        a = G.dominant_orientation_deg(f)
        d = abs(a - real) % 180.0
        errs.append(min(d, 180.0 - d))
    peor = max(errs)
    assert peor < 8.0, f"orientacion de linea fina rota: peor error {peor:.1f} deg, {errs}"
    print("orientacion de linea fina OK: peor error %.1f deg" % peor)


if __name__ == "__main__":
    test_todo()
    test_orientacion_linea_fina()
    print("TODAS LAS COMPROBACIONES PASAN")
