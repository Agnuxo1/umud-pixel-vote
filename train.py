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
def evaluate(model, loader, dev, fg_weight=10.0):
    model.eval()
    inter = np.zeros(VOCAB); union = np.zeros(VOCAB); loss = 0.0; nb = 0
    for x, y, a in loader:
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
    vg = a.val_group or usables[a.seed % len(usables)]
    groups = usables
    va = tr_all[tr_all.grupo == vg]
    tr = tr_all[tr_all.grupo != vg]
    print(f"grupos_usables={len(groups)} validacion={vg} n_tr={len(tr)} n_va={len(va)}", flush=True)

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
    hist, best = [], 1e9
    for ep in range(a.epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        for x, y, al in dl:
            x, y, al = x.to(dev), y.to(dev), al.to(dev)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = masked_ce(model(x), y, al, a.fg_weight)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += float(loss.detach()); nb += 1
        vl, iou = evaluate(model, dv, dev, a.fg_weight)
        rec = dict(epoch=ep, train_loss=tot / max(nb, 1), val_loss=vl,
                   iou_apo_sup=float(iou[1]), iou_apo_deep=float(iou[2]),
                   iou_ori_media=float(np.nanmean(iou[3:])), s=round(time.time() - t0, 1))
        hist.append(rec); print(json.dumps(rec), flush=True)
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "arch": a.arch, "vocab": VOCAB,
                        "steps": a.steps, "seed": a.seed, "val_group": vg}, f"{a.out}/{tag}.pt")
    json.dump(dict(arch=a.arch, seed=a.seed, val_group=vg, best_val=best, hist=hist,
                   params=modelos.n_params(model)), open(f"{a.out}/{tag}.json", "w"), indent=1)
    print("LISTO", tag, "best_val", round(best, 4), flush=True)


if __name__ == "__main__":
    main()
