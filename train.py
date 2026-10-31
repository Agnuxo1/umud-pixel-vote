"""Entrenamiento del campo de votacion por pixel (UMUD).

Etiquetas: el reto no publica PA/FL/MT, solo mascaras. Se construye un vocabulario unico
    0            fondo
    1, 2         aponeurosis superficial / profunda      (supervisado por apo_masks)
    3 .. 3+K-1   bins de orientacion del fasciculo       (supervisado por fasc_masks)
Cada imagen supervisa solo el subconjunto de palabras del que tiene mascara: los logits de
las demas se anulan antes de la entropia cruzada. Asi un unico modelo aprende las dos cosas
sin inventar etiquetas que no existen.

Validacion agrupada por formato de fotograma nativo (Wm,Hm), que es la huella del equipo:
el test contiene un equipo que no esta en entrenamiento, asi que una validacion aleatoria
seria optimista.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

K_ORI = 24                      # bins de orientacion en [-90, 90): 7,5 grados cada uno.
                                # Con 12 bins el error de cuantizacion (hasta 7,5 grados)
                                # ya superaba por si solo la tolerancia oficial de PA (6 grados).
VOCAB = 3 + K_ORI


def orientation_bins(mask, sigma=2.0):
    """Bin de orientacion por pixel de una mascara de fasciculos (tensor de estructura)."""
    import imutil as U
    m = mask.astype(np.float32)
    gx, gy = U.sobel_xy(m)
    jxx = U.gauss(gx * gx, sigma)
    jyy = U.gauss(gy * gy, sigma)
    jxy = U.gauss(gx * gy, sigma)
    ang = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90.0   # direccion de la estructura
    ang = (ang + 90.0) % 180.0 - 90.0
    b = np.floor((ang + 90.0) / (180.0 / K_ORI)).astype(np.int64)
    return np.clip(b, 0, K_ORI - 1)


def apo_classes(mask):
    """1 = aponeurosis superficial, 2 = profunda, por orden de profundidad de cada banda."""
    import imutil as U
    lab, comps = U.components(mask > 0)
    out = np.zeros(mask.shape, np.int64)
    if not comps:
        return out
    ys = [c["cy"] for c in comps]
    mid = 0.5 * (min(ys) + max(ys))
    for c in comps:
        out[lab == c["label"]] = 1 if c["cy"] <= mid else 2
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, root, rows, train=True):
        self.root, self.rows, self.train = root, rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import imutil as U
        r = self.rows[i]
        kind, name = r["kind"], r["name"]
        img = U.imread_gray(f"{self.root}/{kind}/img/{name}.png")
        msk = U.imread_gray(f"{self.root}/{kind}/msk/{name}.png")
        if self.train and np.random.rand() < 0.5:          # espejo horizontal
            img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
        if self.train:                                      # ganancia y contraste (equipos distintos)
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.75, 1.3)
                          + np.random.uniform(-20, 20), 0, 255)
        x = (img.astype(np.float32) / 255.0 - 0.5)[None]
        if kind == "apo":
            y = apo_classes(msk)
            allowed = [0, 1, 2]
        else:
            b = orientation_bins(msk)
            y = np.where(msk > 0, 3 + b, 0)
            allowed = [0] + list(range(3, VOCAB))
        a = np.zeros(VOCAB, np.float32); a[allowed] = 1.0
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(a)


def masked_ce(logits, y, allowed, fg_weight=10.0):
    """Entropia cruzada restringida a las palabras que esa imagen puede supervisar.

    El fondo ocupa ~98 % de los pixeles: sin peso de clase el modelo aprende a decir
    "fondo" en todas partes y los bins de orientacion nunca llegan a activarse (medido en
    el piloto de control: IoU de orientacion 0,000 tras dos epocas). Se pesa el frente.
    """
    neg = (1.0 - allowed)[:, :, None, None] * -1e4
    w = torch.full((logits.shape[1],), float(fg_weight), device=logits.device, dtype=logits.dtype)
    w[0] = 1.0
    return F.cross_entropy(logits + neg, y, weight=w)


@torch.no_grad()
@torch.no_grad()   # sin esto cada forward de validacion construye el grafo entero, que aqui
                   # no se usa nunca: en un servidor de 15,9 GB sin swap es memoria regalada
def evaluate(model, loader, dev, fg_weight=10.0, limite=0):
    model.eval()
    inter = np.zeros(VOCAB); union = np.zeros(VOCAB); loss = 0.0; nb = 0
    for ib, (x, y, a) in enumerate(loader):
        if limite and ib >= limite:
            break
        x, y, a = x.to(dev), y.to(dev), a.to(dev)
        lg = model(x)
        loss += float(masked_ce(lg, y, a, fg_weight)); nb += 1
        p = (lg + (1.0 - a)[:, :, None, None] * -1e4).argmax(1)
        for c in range(VOCAB):
            pi, yi = (p == c), (y == c)
            inter[c] += float((pi & yi).sum()); union[c] += float((pi | yi).sum())
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    model.train()
    return loss / max(nb, 1), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ds")
    ap.add_argument("--arch", default="control", choices=["control", "control_small", "neuropixel", "hybrid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--val-group", default="")
    ap.add_argument("--sel-group", default="")
    # Prueba de humo: corta cada epoca a N lotes para comprobar en segundos, y en CPU,
    # que el reparto, el guardado por ventana y la evaluacion funcionan antes de gastar
    # un turno de cola en veinte entrenamientos.
    ap.add_argument("--limite-lotes", type=int, default=0)
    ap.add_argument("--fg-weight", type=float, default=10.0)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    import pandas as pd
    import modelos
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    meta = pd.read_csv(f"{a.root}/meta.csv")
    tr_all = meta[meta.kind.isin(["apo", "fasc"])].copy()
    tr_all["grupo"] = tr_all.Wm.astype(str) + "x" + tr_all.Hm.astype(str)
    # solo grupos con suficiente masa y con los DOS tipos de mascara, para que la
    # validacion mida de verdad las tres palabras del vocabulario
    tam = tr_all.groupby("grupo").agg(n=("name", "size"), tipos=("kind", "nunique"))
    usables = sorted(tam[(tam.n >= 50) & (tam.tipos == 2)].index)
    if not usables:
        usables = sorted(tam[tam.n >= 50].index) or sorted(tam.index)
    # Leave-TWO-devices-out. El fallo que la auditoria encontro era que el checkpoint se
    # guardaba en el minimo del MISMO grupo que luego se informa: la oscilacion entre epocas
    # de un solo brazo llega a 0,24-0,61 de perdida, frente a diferencias entre brazos de
    # 0,017, asi que la seleccion elegia ruido y cualquier ablacion leida de ahi estaba
    # dominada por el. Ahora se apartan DOS grupos: `sg` para elegir la epoca y `vg` para
    # informar, y `vg` no se evalua ni una vez durante el entrenamiento (lo hace evaluar2.py
    # despues, sobre el checkpoint ya cerrado). Se aparta otro GRUPO y no una fraccion
    # aleatoria porque asi la senal de seleccion es fuera de dominio, igual que el informe.
    vg = a.val_group or usables[a.seed % len(usables)]
    # El grupo de seleccion no puede ser el siguiente de la lista: con ese reparto a dos
    # semillas les tocaba seleccionar sobre 93 y 120 imagenes, que es justo lo contrario de
    # reducir el ruido de epoca. Se toma el grupo MAS GRANDE que no sea el del informe ni el
    # mayor de todos, de modo que el grupo mayor se queda entrenando salvo cuando se informa
    # sobre el, y la senal de seleccion nunca baja de unos cientos de imagenes.
    por_tam = list(tam.loc[usables, "n"].sort_values(ascending=False).index)
    cand = [g for g in por_tam if g != vg and g != por_tam[0]] or [g for g in por_tam if g != vg]
    sg = a.sel_group or cand[0]
    if sg == vg:
        raise SystemExit("el grupo de seleccion y el de informe no pueden ser el mismo")
    groups = usables
    va = tr_all[tr_all.grupo == sg]            # seleccion interna
    tr = tr_all[~tr_all.grupo.isin([vg, sg])]  # entrenamiento
    print(f"grupos_usables={len(groups)} informe={vg} seleccion={sg} "
          f"n_tr={len(tr)} n_sel={len(va)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = modelos.build(a.arch, VOCAB, **({"steps": a.steps} if a.arch in ("neuropixel", "hybrid") else {})).to(dev)
    print(f"arch={a.arch} params={modelos.n_params(model)}", flush=True)
    dl = torch.utils.data.DataLoader(DS(a.root, tr.to_dict("records"), True), batch_size=a.bs,
                                     shuffle=True, num_workers=2, drop_last=True)
    dv = torch.utils.data.DataLoader(DS(a.root, va.to_dict("records"), False), batch_size=a.bs,
                                     shuffle=False, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, a.epochs * len(dl)))
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.arch}_s{a.seed}"
    # Ventana de 3 epocas: se elige la epoca cuya media movil de perdida de seleccion es
    # minima, no la epoca individual minima, porque el ruido entre epocas es un orden de
    # magnitud mayor que el efecto que se quiere medir. Se guarda la epoca CENTRAL de la
    # ventana ganadora, asi que hay que conservar sus pesos en RAM (en CPU) mientras pasa.
    hist, best = [], 1e9
    ventana, pesos_ventana = [], []
    for ep in range(a.epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        for ib, (x, y, al) in enumerate(dl):
            if a.limite_lotes and ib >= a.limite_lotes:
                break
            x, y, al = x.to(dev), y.to(dev), al.to(dev)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = masked_ce(model(x), y, al, a.fg_weight)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += float(loss.detach()); nb += 1
        vl, iou = evaluate(model, dv, dev, a.fg_weight, limite=a.limite_lotes)
        rec = dict(epoch=ep, train_loss=tot / max(nb, 1), val_loss=vl,
                   iou_apo_sup=float(iou[1]), iou_apo_deep=float(iou[2]),
                   iou_ori_media=float(np.nanmean(iou[3:])), s=round(time.time() - t0, 1))
        hist.append(rec); print(json.dumps(rec), flush=True)
        ventana.append(vl)
        pesos_ventana.append({k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
        if len(ventana) > 3:
            ventana.pop(0); pesos_ventana.pop(0)
        if len(ventana) == 3:
            m = float(np.mean(ventana))
            if m < best:
                best = m
                torch.save({"model": pesos_ventana[1], "arch": a.arch, "vocab": VOCAB,
                            "steps": a.steps, "seed": a.seed, "val_group": vg, "sel_group": sg,
                            "epoca_central": ep - 1, "sel_loss_media3": m},
                           f"{a.out}/{tag}.pt")
    json.dump(dict(arch=a.arch, seed=a.seed, val_group=vg, sel_group=sg,
                   best_sel_media3=best, hist=hist,
                   params=modelos.n_params(model)), open(f"{a.out}/{tag}.json", "w"), indent=1)
    print("LISTO", tag, "mejor_media3_seleccion", round(best, 4),
          "informe_pendiente_en", vg, flush=True)


if __name__ == "__main__":
    main()
