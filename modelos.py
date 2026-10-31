"""Dos modelos de campo por pixel para UMUD, con la MISMA cabeza de salida.

- `Control`: U-Net clasica (codificador convolucional con submuestreo y decodificador con
  saltos). Es el control profesional del experimento.
- `NeuroPixelField`: adaptacion fiel de NeuroPixel (D:/PROJECTS/196_NeuroPixel) a un campo
  de votacion denso. Conserva sus tres piezas:
    1. `Retina`: convoluciones dilatadas (1,2,4,8), campo receptivo ~31 px y **sin reducir
       resolucion** -> cada pixel recibe un "color" de c_id canales (identidad).
    2. Automata celular neuronal: `perceive` 3x3 depthwise + MLP 1x1 residual con
       `fire_rate`, reinyectando la identidad en cada paso. Cada pixel solo ve a sus 8
       vecinos; la estructura de linea se propaga por iteracion, no por submuestreo.
    3. Lectura con **diccionario compartido atado**: el mismo `read @ E^T` se aplica a
       TODOS los pixeles (`lens_logits` en el original). Eso es, literalmente, que cada
       pixel vote una palabra del diccionario.
  Vocabulario aqui: {fondo, aponeurosis superficial, aponeurosis profunda} + K bins de
  orientacion de fasciculo.

Ambos devuelven logits B x V x H x W, de modo que la etapa geometrica y la ablacion son
identicas y la unica variable es el cuerpo del modelo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Retina(nn.Module):
    """Igual que neuropixel/model.py: convoluciones dilatadas sin perder resolucion."""

    def __init__(self, c_in=1, c_id=16, w=32):
        super().__init__()
        layers, cin = [], c_in
        for d in (1, 2, 4, 8):
            layers += [nn.Conv2d(cin, w, 3, padding=d, dilation=d), nn.ReLU()]
            cin = w
        layers.append(nn.Conv2d(w, c_id, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuroPixelField(nn.Module):
    def __init__(self, vocab, c_id=16, c=48, hidden=128, steps=16, fire_rate=0.5):
        super().__init__()
        self.retina = Retina(1, c_id)
        self.steps, self.fire_rate = steps, fire_rate
        self.embed = nn.Embedding(vocab, c_id)          # diccionario (pesos atados)
        self.seed = nn.Conv2d(c_id, c, 1)
        self.perceive = nn.Conv2d(c, 2 * c, 3, padding=1, groups=c, bias=False)
        self.f1 = nn.Conv2d(3 * c + c_id, hidden, 1)
        self.f2 = nn.Conv2d(hidden, c, 1)
        nn.init.zeros_(self.f2.weight); nn.init.zeros_(self.f2.bias)
        self.read = nn.Linear(c, c_id)

    def lens_logits(self, s):
        """El diccionario aplicado a TODOS los pixeles: B,C,H,W -> B,V,H,W."""
        v = self.read(s.permute(0, 2, 3, 1)) @ self.embed.weight.T
        return v.permute(0, 3, 1, 2)

    def forward(self, x, steps=None):
        ids = self.retina(x)                             # identidad por pixel
        s = self.seed(ids)
        for _ in range(steps or self.steps):
            h = torch.cat([s, self.perceive(s), ids], 1)
            ds = self.f2(F.relu(self.f1(h)))
            if self.training and self.fire_rate < 1:
                ds = ds * (torch.rand_like(ds[:, :1]) < self.fire_rate)
            s = s + ds
        return self.lens_logits(s)


def _blk(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class Control(nn.Module):
    """U-Net de referencia. Si `segmentation_models_pytorch` esta disponible se usa su
    U-Net ResNet-34, que es la herramienta profesional estandar para este problema."""

    def __init__(self, vocab, width=32, smp=True):
        super().__init__()
        self.smp = None
        if smp:
            try:
                import segmentation_models_pytorch as S
                self.smp = S.Unet("resnet34", encoder_weights=None, in_channels=1, classes=vocab)
            except Exception as exc:
                # El respaldo NO puede ser silencioso: cambia el brazo de la comparacion sin
                # avisar. Paso asi: la libreria no estaba en el servidor, se entrenó la U-Net
                # propia de 1,95 M y se informo como "control profesional ResNet-34", que son
                # ~24 M. Quien lea una tabla tiene derecho a saber que cuerpo corrio.
                print(f"AVISO modelos.Control: sin segmentation_models_pytorch ({exc.__class__.__name__}): "
                      f"se usa la U-Net propia, NO un ResNet-34", flush=True)
                self.smp = None
        self.cuerpo = "smp_unet_resnet34" if self.smp is not None else "unet_propia"
        if self.smp is None:
            w = width
            self.e1, self.e2, self.e3, self.e4 = _blk(1, w), _blk(w, 2 * w), _blk(2 * w, 4 * w), _blk(4 * w, 8 * w)
            self.d3, self.d2, self.d1 = _blk(12 * w, 4 * w), _blk(6 * w, 2 * w), _blk(3 * w, w)
            self.out = nn.Conv2d(w, vocab, 1)
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.smp is not None:
            return self.smp(x)
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        u = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u, e3], 1))
        u = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u, e2], 1))
        u = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.d1(torch.cat([u, e1], 1)))


class Hybrid(nn.Module):
    """Control + capa NeuroPixel encima: la ablacion mide exactamente esa capa."""

    def __init__(self, vocab, steps=8, **kw):
        super().__init__()
        self.base = Control(vocab, **kw)
        self.np = NeuroPixelField(vocab, steps=steps)
        self.mix = nn.Conv2d(2 * vocab, vocab, 1)
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)

    def forward(self, x):
        a = self.base(x)
        b = self.np(x)
        return a + self.mix(torch.cat([a, b], 1))


def build(arch, vocab, **kw):
    """`control` = U-Net profesional; `control_small` = misma U-Net con la capacidad del
    modelo propio, para que la comparacion no la gane el numero de parametros."""
    if arch == "control_small":
        kw = {k: v for k, v in kw.items() if k != "steps"}
        return Control(vocab, width=8, smp=False, **kw)
    if arch == "control":
        kw = {k: v for k, v in kw.items() if k != "steps"}
        return Control(vocab, **kw)
    return {"neuropixel": NeuroPixelField, "hybrid": Hybrid}[arch](vocab, **kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
