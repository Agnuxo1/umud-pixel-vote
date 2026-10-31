"""Geometria de arquitectura muscular: de campos de pixel a PA, MT y FL.

Entrada (en pixeles de la imagen distribuida):
  - apo: probabilidad por pixel de aponeurosis (0-1)
  - fasc: probabilidad por pixel de fasciculo (0-1), o un campo de orientacion
  - sx, sy: pixeles por milimetro en cada eje
Salida: pa_deg, fl_mm, mt_mm segun el procedimiento de anotacion del reto:
  - MT  = distancia perpendicular entre aponeurosis superficial y profunda, promediada
          en TRES posiciones repartidas por el ancho del musculo.
  - PA  = angulo entre la linea de fasciculo y la aponeurosis PROFUNDA.
  - FL  = distancia entre aponeurosis a lo largo de la linea de fasciculo; con fasciculos
          que salen del encuadre se extrapola linealmente -> FL = MT / sin(PA).
Nada aqui esta calibrado contra el leaderboard.
"""
from __future__ import annotations

import numpy as np


def aponeurosis_curves(apo, thr=0.5, min_rel_extent=0.30, deg=2):
    """Separa el mapa de aponeurosis en bandas y ajusta una curva y(x) a cada una.

    Devuelve lista de (coef_poly, x_min, x_max, peso) ordenada por profundidad media.
    """
    import imutil as U
    bw = apo > thr
    if bw.sum() == 0:
        return []
    bw = U.close_rect(bw, 5, 25)
    lab, comps = U.components(bw)
    H, W = apo.shape
    out = []
    for c in comps:
        if c["w"] < min_rel_extent * W:
            continue
        ys, xs = np.where(lab == c["label"])
        # linea central de la banda: y medio por columna
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        xu = np.unique(xs)
        ym = np.array([ys[xs == x].mean() for x in xu])
        if xu.size < max(10, deg + 1):
            continue
        cf = np.polyfit(xu, ym, deg)
        out.append((cf, float(xu.min()), float(xu.max()), float(len(ys))))
    out.sort(key=lambda t: np.polyval(t[0], 0.5 * (t[1] + t[2])))
    return out


def _normal_distance_mm(c_sup, c_deep, x, sx, sy):
    """Distancia perpendicular en mm de la curva superficial a la profunda en x.

    Se mide a lo largo de la normal a la aponeurosis profunda, en espacio metrico.
    """
    m = np.polyval(np.polyder(c_deep), x) * (sx / sy)      # pendiente en mm/mm
    ysup_mm = np.polyval(c_sup, x) / sy
    ydeep_mm = np.polyval(c_deep, x) / sy
    dv = ydeep_mm - ysup_mm                                 # separacion vertical (mm)
    return abs(dv) / np.sqrt(1.0 + m * m)


def thickness_mm(curves, sx, sy, n_sites=3):
    """MT: media de la distancia perpendicular en tres posiciones del ancho comun."""
    if len(curves) < 2:
        return np.nan, None, None
    sup, deep = curves[0], curves[-1]
    x0 = max(sup[1], deep[1]); x1 = min(sup[2], deep[2])
    if x1 - x0 < 10:
        return np.nan, sup, deep
    xs = np.linspace(x0 + 0.1 * (x1 - x0), x1 - 0.1 * (x1 - x0), n_sites)
    d = [_normal_distance_mm(sup[0], deep[0], x, sx, sy) for x in xs]
    return float(np.mean(d)), sup, deep


def deep_slope_deg(c_deep, x, sx, sy):
    """Inclinacion de la aponeurosis profunda en grados, en espacio metrico."""
    m = np.polyval(np.polyder(c_deep), x) * (sx / sy)
    return float(np.degrees(np.arctan(m)))


def fascicle_angle_deg(theta_px, sx, sy):
    """Convierte un angulo medido en pixeles a angulo metrico.

    theta_px: angulo respecto al eje x en la imagen (grados). Al ser el pixel anisotropo,
    tan(theta_mm) = tan(theta_px) * (sx/sy).
    """
    t = np.tan(np.radians(theta_px)) * (sx / sy)
    return float(np.degrees(np.arctan(t)))


def pennation_and_length(theta_fasc_px, c_deep, x_ref, mt_mm, sx, sy,
                         pa_lo=5.0, pa_hi=45.0, fl_lo=30.0, fl_hi=200.0):
    """PA = |angulo del fasciculo - angulo de la aponeurosis profunda| (en mm).
    FL = MT / sin(PA) (extrapolacion lineal, el procedimiento del reto)."""
    a_f = fascicle_angle_deg(theta_fasc_px, sx, sy)
    a_d = deep_slope_deg(c_deep, x_ref, sx, sy) if c_deep is not None else 0.0
    pa = abs(a_f - a_d)
    pa = float(np.clip(pa, pa_lo, pa_hi))
    fl = mt_mm / np.sin(np.radians(pa)) if np.isfinite(mt_mm) and pa > 0 else np.nan
    if np.isfinite(fl):
        fl = float(np.clip(fl, fl_lo, fl_hi))
    return pa, fl


def dominant_orientation_deg(fasc, weights=None, nbins=180):
    """Orientacion dominante de un mapa de fasciculos por tensor de estructura.

    Devuelve el angulo en grados respecto al eje x de la imagen (positivo hacia abajo).
    """
    import imutil as U
    f = fasc.astype(np.float32)
    gx, gy = U.sobel_xy(f)
    # NO ponderar por f. Las mascaras de fasciculo son lineas de 2 px de grosor medio, y en
    # una linea tan fina el gradiente de Sobel es ~0 SOBRE la propia linea: toda la energia
    # esta en los vecinos, donde f=0. Pesar por f la borraba justo donde esta. Medido con
    # lineas de respuesta conocida: el error medio era 54,7 deg (una linea horizontal daba
    # -90 deg, el maximo posible) y sin pesar es 1,6 deg.
    # `weights` sigue disponible para mapas de probabilidad, donde el fondo SI tiene gradiente
    # y conviene acotar la region; para una mascara binaria no hace falta, porque el fondo
    # plano no contribuye.
    w = 1.0 if weights is None else weights
    jxx = float((gx * gx * w).sum()); jyy = float((gy * gy * w).sum()); jxy = float((gx * gy * w).sum())
    # 0.5*atan2 da la direccion del GRADIENTE; la estructura es perpendicular: +90 grados
    ang = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90.0
    return float((ang + 90.0) % 180.0 - 90.0)   # a [-90, 90)


def radon_orientation_deg(img, mask=None, lo=-60, hi=60, step=0.5):
    """Orientacion por transformada de Radon: el angulo cuya proyeccion es mas picuda.

    Control clasico frente al campo de votacion por pixel.
    """
    import imutil as U
    a = img.astype(np.float32)
    if mask is not None:
        a = a * mask
    thetas = np.arange(lo, hi + 1e-9, step)
    var = U.radon_variance(a, thetas)
    return float(thetas[int(np.argmax(var))])
