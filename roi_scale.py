"""Escala fisica (px/mm) de las ecografias UMUD a partir de las reglas del equipo.

Sin OCR y sin ninguna constante ajustada al leaderboard. El metodo:
 1. se busca en TODA la imagen la banda de columnas (o de filas) cuyo perfil de maximos
    contiene el tren de marcas mas regular: esa es la regla del ecografo;
 2. el periodo se estima como el espaciado minimo consistente, de modo que una marca que
    falte produzca un hueco multiplo y no desvirtue la mediana;
 3. el valor fisico de la marca se elige entre 2, 5 y 10 mm: el que deja la profundidad
    cubierta por la regla dentro del rango fisiologico de estas adquisiciones.

Prior art publica citada: cuaderno de watehell "ans is simple: physical scale + U-Net +
geometry" (Kaggle 2026), que recupera la escala por OCR de la profundidad mas marcas de
regla con cuatro disposiciones de equipo fijas. Aqui no se usa OCR ni disposiciones fijas:
la regla se localiza por regularidad y el valor de marca se decide por rango fisiologico.
"""
from __future__ import annotations

import numpy as np

TICK_CANDIDATES_MM = (2.0, 5.0, 10.0)
DEPTH_RANGE_MM = (20.0, 110.0)      # profundidad plausible del campo de vision
WIDTH_RANGE_MM = (20.0, 120.0)      # anchura plausible de una sonda lineal


def load_gray(path):
    """cv2 en vez de PIL: el libtiff de Anaconda aborta el proceso con estos .tif."""
    import cv2
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise OSError(f"no se pudo leer {path}")
    return g.astype(np.float32)


def peaks_1d(prof, rel=0.40, min_sep=4, max_width=6):
    """Centros de picos ESTRECHOS y brillantes de un perfil 1-D.

    Las marcas de regla son finas (pocos pixeles). Los renglones de texto y las barras de
    color dan picos anchos: se descartan por anchura, que es lo que separa una regla de
    la columna de texto del ecografo."""
    p = prof - np.median(prof)
    if p.size == 0 or p.max() <= 0:
        return np.array([]), 0.0
    thr = max(2.0, rel * float(p.max()))
    idx = np.where(p > thr)[0]
    if idx.size == 0:
        return np.array([]), 0.0
    ink = float(idx.size) / float(p.size)
    if ink > 0.25:                      # banda saturada: ni se agrupa
        return np.array([]), ink
    cuts = np.where(np.diff(idx) > 2)[0] + 1
    groups = np.split(idx, cuts)
    ok = np.array([g.size <= max_width for g in groups])
    if not ok.any():
        return np.array([]), ink
    c = np.array([g.mean() for g, k in zip(groups, ok) if k])
    sep = np.concatenate(([min_sep + 1.0], np.diff(c)))
    return c[sep >= min_sep], ink


def train_period(c, tol=0.18):
    """Periodo de un tren de marcas con huecos: el espaciado minimo del que los demas
    son multiplos enteros. Devuelve (periodo, n_marcas_explicadas, cobertura)."""
    if c.size < 4:
        return np.nan, 0, 0.0
    d = np.diff(c)
    d = d[d > 2]
    if d.size < 3:
        return np.nan, 0, 0.0
    base = float(np.median(d[d <= np.percentile(d, 60)]))   # espaciado "corto" tipico
    if not np.isfinite(base) or base < 4:
        return np.nan, 0, 0.0
    k = np.round(d / base)
    ok = (k >= 1) & (np.abs(d - k * base) <= tol * base)
    if ok.mean() < 0.7:
        return np.nan, 0, float(ok.mean())
    per = float(np.sum(d[ok]) / np.sum(k[ok]))              # ajuste por minimos cuadrados
    return per, int(ok.sum() + 1), float(ok.mean())


def scan_ruler_multi(g, axis, wins=(3, 6, 10, 18), step=4):
    """Barrido multiescala: las reglas van de 3 px (borde de un recorte) a ~18 px."""
    best = None
    for w in wins:
        b = scan_ruler(g, axis, win=w, step=step)
        if b and (best is None or b["score"] > best["score"]):
            best = b
    return best


def scan_ruler(g, axis, win=20, step=10):
    """Recorre bandas de la imagen y devuelve la regla mas regular encontrada.

    axis='y': bandas de columnas, marcas apiladas en vertical  -> escala vertical.
    axis='x': bandas de filas,    marcas alineadas en horizontal -> escala horizontal.
    """
    from scipy.ndimage import maximum_filter1d
    H, W = g.shape
    n = W if axis == "y" else H
    # maximo deslizante: el perfil de CADA banda de anchura `win` de una sola pasada
    M = maximum_filter1d(g, size=win, axis=1 if axis == "y" else 0, mode="nearest")
    best = None
    for a in range(0, max(1, n - win), step):
        ctr = min(n - 1, a + win // 2)
        prof = M[:, ctr] if axis == "y" else M[ctr, :]
        c, ink = peaks_1d(prof)
        if ink > 0.12:            # banda con mucha tinta: texto o imagen, no regla
            continue
        per, nmark, cov = train_period(c)
        if not np.isfinite(per) or nmark < 6 or cov < 0.85:
            continue
        # Las marcas de una regla tienen casi el mismo brillo y la misma anchura; los
        # renglones de texto del ecografo, no. Esto separa la regla de la columna de datos.
        vals = prof[np.clip(np.round(c).astype(int), 0, len(prof) - 1)]
        unif = float(np.std(vals) / (np.mean(vals) + 1e-6))
        if unif > 0.35:
            continue
        span = float(c[-1] - c[0])
        score = nmark + 6.0 * cov + 2.0 * (1.0 - unif) + 2.0 * span / max(len(prof), 1)
        if best is None or score > best["score"]:
            best = dict(score=score, period=per, n=nmark, cov=cov, band=(a, a + win),
                        first=float(c[0]), last=float(c[-1]), span=span, ink=ink)
    return best


def choose_tick_mm(span_px, period_px, rng):
    """Valor fisico de la marca entre 2/5/10 mm segun el rango fisiologico del tramo."""
    lo, hi = rng
    mid = 0.5 * (lo + hi)
    best, bestpen = np.nan, np.inf
    for t in TICK_CANDIDATES_MM:
        ext = span_px / period_px * t
        pen = (0.0 if lo <= ext <= hi else min(abs(ext - lo), abs(ext - hi))) + 0.02 * abs(ext - mid)
        if pen < bestpen:
            best, bestpen = t, pen
    return best


def measure(path):
    g = load_gray(path)
    H, W = g.shape
    ry = scan_ruler_multi(g, "y")
    rx = scan_ruler_multi(g, "x")
    r = dict(path=str(path), H=H, W=W, sx=np.nan, sy=np.nan, tick_y=np.nan, tick_x=np.nan,
             ny=0, nx=0, span_y=np.nan, span_x=np.nan, y0=np.nan, y1=np.nan,
             band_y=np.nan, band_x=np.nan, cov_y=np.nan, cov_x=np.nan)
    if ry:
        t = choose_tick_mm(ry["span"], ry["period"], DEPTH_RANGE_MM)
        ext = ry["span"] / ry["period"] * t
        if DEPTH_RANGE_MM[0] <= ext <= DEPTH_RANGE_MM[1]:   # si no, no era una regla
            r.update(sy=ry["period"] / t, tick_y=t, ny=ry["n"], span_y=ry["span"],
                     y0=ry["first"], y1=ry["last"], band_y=ry["band"][0], cov_y=ry["cov"])
    if rx:
        t = choose_tick_mm(rx["span"], rx["period"], WIDTH_RANGE_MM)
        ext = rx["span"] / rx["period"] * t
        if WIDTH_RANGE_MM[0] <= ext <= WIDTH_RANGE_MM[1]:
            r.update(sx=rx["period"] / t, tick_x=t, nx=rx["n"], span_x=rx["span"],
                     band_x=rx["band"][0], cov_x=rx["cov"])
    # una sola regla: se asume pixel isotropo en el marco nativo (crops y recortes)
    if np.isnan(r["sy"]) and np.isfinite(r["sx"]):
        r["sy"] = r["sx"]; r["iso"] = "sx->sy"
    elif np.isnan(r["sx"]) and np.isfinite(r["sy"]):
        r["sx"] = r["sy"]; r["iso"] = "sy->sx"
    else:
        r["iso"] = ""
    r["depth_mm"] = r["span_y"] / r["sy"] if np.isfinite(r["sy"]) else np.nan
    return r


if __name__ == "__main__":
    import glob, sys
    import pandas as pd
    pats = sys.argv[1:] or ["E:/UMUD-data/raw/test_images_v2/test_set_v2/*.tif",
             "E:/UMUD-data/raw/test_images_v2/test_set_v2/*.png"]
    files = [f for p in pats for f in sorted(glob.glob(p))]
    df = pd.DataFrame([measure(f) for f in files])
    out = "E:/UMUD-work/scale_test.csv"
    df.to_csv(out, index=False)
    print(f"n={len(df)}  sy={df.sy.notna().mean():.3f}  sx={df.sx.notna().mean():.3f} -> {out}")
    print(df.groupby([df.W, df.H]).agg(n=("sy", "size"), sy=("sy", "median"), sx=("sx", "median"),
                                       ticky=("tick_y", "median"), tickx=("tick_x", "median")).round(2))


# --------------------------------------------------------------------------------------
# Consolidacion por grupo: la mejora propia sobre la prior art
# --------------------------------------------------------------------------------------
def consolidar(df, iso_lo=0.6, iso_hi=1.8):
    """Reconcilia las escalas de un conjunto usando dos invariantes fisicos.

    1. Dentro de un mismo formato de imagen (W,H) el equipo y su reescalado son los mismos,
       asi que la ANISOTROPIA sx/sy es comun: se toma su mediana entre las imagenes en las
       que las dos reglas se detectaron con el mismo valor de marca y cobertura total.
    2. Una escala vertical aislada que se aparte mucho de la mediana de su grupo procede de
       una regla mal leida: se sustituye por la mediana del grupo.
    Ninguna de las dos reglas mira al leaderboard: son coherencia interna del conjunto.
    """
    import numpy as np
    import pandas as pd
    df = df.copy()
    df["grupo"] = list(zip(df.W, df.H))
    fiable = (df.cov_y == 1.0) & (df.cov_x == 1.0) & (df.tick_y == df.tick_x) & df.sx.notna() & df.sy.notna()
    ratio = (df.sx / df.sy).where(fiable)
    aniso = ratio.groupby(df.grupo).median()
    med_sy = df.sy.groupby(df.grupo).median()
    for g in df.grupo.unique():
        m = df.grupo == g
        a = aniso.get(g, np.nan)
        s0 = med_sy.get(g, np.nan)
        # 2: escala vertical atipica -> mediana del grupo
        if np.isfinite(s0):
            mala = m & (df.sy.isna() | (df.sy / s0 > 1.6) | (df.sy / s0 < 0.625))
            df.loc[mala, ["sy", "sy_src"]] = [s0, "mediana_grupo"]
        # 1: anisotropia comun
        if np.isfinite(a) and iso_lo <= a <= iso_hi:
            malx = m & (df.sx.isna() | ((df.sx / df.sy) / a > 1.3) | ((df.sx / df.sy) / a < 0.77))
            df.loc[malx, "sx"] = df.loc[malx, "sy"] * a
            df.loc[malx, "sx_src"] = "aniso_grupo"
        else:
            malx = m & (df.sx.isna() | (df.sx / df.sy > iso_hi) | (df.sx / df.sy < iso_lo))
            df.loc[malx, "sx"] = df.loc[malx, "sy"]      # pixel isotropo como ultimo recurso
            df.loc[malx, "sx_src"] = "isotropo"
    df["depth_mm"] = df.span_y / df.sy
    return df


def huella_equipo(df):
    """Huella del equipo: formato de imagen + donde esta la regla + valor de marca + tramo.

    JEV eligio `learned_scale` para rellenar las imagenes sin regla, pero con confianza 0,39;
    se desempata con evidencia: el grupo (W,H) por si solo YA demostro ser incorrecto para el
    formato 1200x800 (E03 del registro), y la causa medida es que ese formato mezcla varios
    equipos. La huella anade justo lo que los distingue y no necesita entrenar nada.
    """
    import numpy as np
    b = (df.band_y / 100).round().fillna(-1)
    t = df.tick_y.fillna(-1)
    s = (df.span_y / 25).round().fillna(-1)
    return list(zip(df.W, df.H, b, t, s))


def consolidar_huella(df, min_n=3):
    """Rellena y corrige la escala con la mediana de su huella; si la huella es demasiado
    pequena, se deja sin escala y la geometria usa su alternativa declarada."""
    import numpy as np
    df = df.copy()
    df["huella"] = huella_equipo(df)
    df["sy_src"] = np.where(df.sy.notna(), "regla", "")
    df["sx_src"] = np.where(df.sx.notna(), "regla", "")
    g = df.groupby("huella").sy
    med, cnt = g.median(), g.count()
    for h in df.huella.unique():
        m = df.huella == h
        if cnt.get(h, 0) >= min_n and np.isfinite(med.get(h, np.nan)):
            malo = m & (df.sy.isna() | (df.sy / med[h] > 1.6) | (df.sy / med[h] < 0.625))
            df.loc[malo, "sy"] = med[h]
            df.loc[malo, "sy_src"] = "huella"
    # anisotropia: solo cuando las dos reglas coinciden en valor de marca
    fiable = (df.tick_y == df.tick_x) & df.sx.notna() & df.sy.notna() & (df.cov_y == 1.0) & (df.cov_x == 1.0)
    ratio = (df.sx / df.sy).where(fiable).groupby(df.huella).median()
    for h in df.huella.unique():
        m = df.huella == h
        a = ratio.get(h, np.nan)
        if np.isfinite(a) and 0.6 <= a <= 1.8:
            malx = m & (df.sx.isna() | ((df.sx / df.sy) / a > 1.3) | ((df.sx / df.sy) / a < 0.77))
            df.loc[malx, "sx"] = df.loc[malx, "sy"] * a
            df.loc[malx, "sx_src"] = "aniso_huella"
        else:
            malx = m & (df.sx.isna() | (df.sx / df.sy > 1.8) | (df.sx / df.sy < 0.6))
            df.loc[malx, "sx"] = df.loc[malx, "sy"]
            df.loc[malx, "sx_src"] = "isotropo"
    return df
