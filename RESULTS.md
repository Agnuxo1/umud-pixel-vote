# Results

Everything here was measured on this machine and on a private RTX 3090 server; nothing was
calibrated against the Kaggle public leaderboard. Negative results are kept.

## 1. The official metric, reproduced

Taken from the organizers' own scoring notebook and executed locally:

```
S = (1/3) * sum_c MAE_c / tau_c  +  1e-6 * (1/3) * sum_c MedAE_c / tau_c
                                 +  1e-9 * (1/3) * sum_c RMSE_c  / tau_c
tau = { pa_deg: 6.0 deg, fl_mm: 12.0 mm, mt_mm: 3.0 mm },  equal weights, per image
```

## 2. Why physical scale dominates — measured, not assumed

The released images are resized while the masks keep the native frame size, so the pixel is
anisotropic. Over the 3809 training images the true anisotropy `(Wi/Wm)/(Hi/Hm)` ranges from
**0.835 to 1.264** (median 1.000). Ignoring it misstates angles and lengths by up to 26 %.

## 3. Hard validation of the OCR-free scale estimator

`roi_scale.py` never sees the masks, so the native frame size is an independent ground truth
for it.

| check | result |
|---|---|
| Depth recovered on a frame whose scanner printed "4.5 cm" | **45.0 mm** |
| Per-image ruler read, training aponeurosis set | 93.9 % of 1048 |
| Per-image ruler read, training fascicle set | 98.3 % of 2761 |
| Per-image ruler read, **test set** | 80.9 % of 309 |
| Measured vs true anisotropy, images with both rulers reliable (n = 148) | median relative error **0.47 %**; within 5 % for **85.1 %**; within 10 % for 85.8 % |
| Test coverage after device-fingerprint consolidation | **98.7 %** (4 images of 309 left without a scale) |

## 4. Negative results kept

- **Consolidating the scale by image-size group (W, H) is wrong.** It raises coverage to
  99.4 % but the 1200x800 bucket holds 239 of the 309 test images and mixes several devices
  and depth settings (sy from 5.3 to 13.5 px/mm), so the group median is wrong for most of
  them. Replaced by a device fingerprint (image format + ruler band position + tick value +
  ruler span).
- **Twelve orientation bins are not enough.** A synthetic geometry test showed a quantisation
  error of up to 7.5 deg, above the official PA tolerance of 6 deg. Raised to 24 bins, with a
  probability-weighted circular mean interpolating inside the bin.
- **Unweighted cross-entropy never learns fascicle orientation.** In a 3-epoch control pilot
  on the server the orientation IoU stayed at 0.000 while aponeurosis IoU reached 0.42,
  because background covers about 98 % of the pixels. Foreground class weighting was added.

## 5. Geometry stage, verified against a constructed case

`test_geometria.py` builds a field whose answer is known: two aponeuroses 120 px apart at
6 px/mm, fascicles at 20 deg.

| quantity | expected | obtained |
|---|---|---|
| MT | 20.00 mm | **20.00 mm** |
| PA | 20 deg | **18.75 deg** (within one 7.5 deg bin) |
| FL | MT / sin(PA) | consistent |
| angle under sx = 2 sy | larger | 20.0 deg -> 36.05 deg |

## 6. Model comparison

In progress on the server: four bodies (professional U-Net control 1 949 979 params,
capacity-matched small control 122 907, `NeuroPixelField` **58 288**, hybrid 2 009 752) x 3
seeds, validated on a held-out device group. Results will be added here per parameter with
seeds and spread, together with the ablation of the own layer. **No result is claimed for
NeuroPixel until it has been measured.**
