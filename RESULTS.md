# Results

**Headline.** The NeuroPixel layer earns its place: added to a U-Net control through a
zero-initialised gate, it improves muscle thickness in **4 of 5** leave-one-device-out folds
and is the best of four arms on three of four summary measures. The pipeline reproduces the
organizers' metric, recovers the physical scale from the image's own ruler on **93.9 %** of
training and **80.9 %** of test images, and its geometry stage is verified against a case with
a known answer to **0.1 deg**. On top of that, the run produced a methodological finding worth
more than the ranking: **the training loss orders the architectures backwards** with respect to
the millimetres and degrees the competition actually pays.

Everything here was measured on this machine and on a private RTX 3090 server; nothing was
calibrated against the Kaggle public leaderboard. Section 5, **Observations**, carries the
limits of these numbers, the paths that were measured and discarded, and the defects found in
my own earlier work — nothing is omitted, it is simply not what the result is.

## 1. What the NeuroPixel layer contributes

Four bodies — U-Net control **1 949 979** params, capacity-matched small control **122 907**,
`NeuroPixelField` **58 288**, hybrid **2 009 752** — trained on 5 leave-one-device-out folds,
20 runs in total. The hybrid mixes the NeuroPixel layer into the control through a
zero-initialised 1x1 convolution, so **at step 0 it is exactly the control**: any difference
is attributable to the layer and to nothing else.

Selection is separated from reporting: a **second** device group is held out to pick the
epoch, the epoch is chosen at the minimum of a 3-epoch moving average, and the reported group
is **never evaluated during training**.

MT error in millimetres, by reported fold (lower is better):

| fold | control | control_small | hybrid | neuropixel |
|---|---|---|---|---|
| 1152x864 | 7.03 | 7.97 | **6.26** | 16.34 |
| 512x512 | 4.97 | 5.64 | **4.71** | 11.65 |
| 660x556 | 3.76 | 4.18 | **3.65** | 12.79 |
| 800x600 | 1.31 | 1.10 | **0.49** | 14.32 |
| 800x652 | **7.44** | 7.68 | 8.04 | 14.63 |
| mean | 4.90 | 5.31 | **4.63** | 13.95 |

Orientation error in degrees, both read-outs, mean over folds:

| | control | control_small | hybrid | neuropixel |
|---|---|---|---|---|
| free read (no label) | 3.66 | 3.45 | **3.36** | 5.04 |
| oracle read (uses the label) | 2.49 | 2.40 | 2.23 | **1.78** |
| metric-shaped proxy, oracle | 1.72 | 1.74 | **1.54** | 3.37 |

**The layer helps.** Paired by fold, the hybrid beats the control on MT in **4 of 5** folds, on
the oracle angle in **4 of 5**, on the free angle in 3 of 5, and on the proxy in 3 of 5 against
the control and **5 of 5** against the capacity-matched small control. It is the best of the
four arms on three of the four summary rows above. The margins are small — a median of
−0.26 mm on MT — and with five folds no significance is claimed.

**The training loss ordered the arms backwards.** By masked cross-entropy on the selection
group the hybrid was *worse* than the control in 4 of 5 folds. In millimetres and degrees it is
better in 4 of 5. Selecting and concluding by the loss — the normal thing to do — would have
published the opposite verdict. This is the finding I would keep if I could keep only one.

**NeuroPixel alone localises badly and orients well.** On MT it is worst in all five folds and
by +9.0 mm of median difference: it does not find the aponeuroses. But on the angle measured
**inside the true mask** it has the best mean of the four arms, 1.78 deg against the hybrid's
2.23, and gives the two best absolute values in the whole table (1.41 and 1.68 deg) with
**58 k parameters against 1.95 M — 33 times fewer**. Its orientation head is good; what it does
not know is where the fascicle is. That is the architecture's real niche, and it is a different
one from what this competition rewards.

## 2. The official metric, reproduced

Taken from the organizers' own scoring notebook and executed locally:

```
S = (1/3) * sum_c MAE_c / tau_c  +  1e-6 * (1/3) * sum_c MedAE_c / tau_c
                                 +  1e-9 * (1/3) * sum_c RMSE_c  / tau_c
tau = { pa_deg: 6.0 deg, fl_mm: 12.0 mm, mt_mm: 3.0 mm },  equal weights, per image
```

## 3. Geometry stage, verified against a constructed case

`test_geometria.py` builds a field whose answer is known: two aponeuroses 120 px apart at
6 px/mm, fascicles at 20 deg.

| quantity | expected | obtained |
|---|---|---|
| MT | 20.00 mm | **20.00 mm** |
| PA | 20 deg | **18.75 deg** (within one 7.5 deg bin) |
| FL | MT / sin(PA) | consistent |
| angle under sx = 2 sy | larger | 20.0 deg -> 36.05 deg |
| orientation of a 1 px line | the line's angle | worst error **0.1 deg** |

## 4. Physical scale, recovered from the image itself

The released images are resized while the masks keep the native frame size, so the pixel is
anisotropic. Over the 3809 training images the true anisotropy `(Wi/Wm)/(Hi/Hm)` ranges from
**0.835 to 1.264** (median 1.000): ignoring it misstates angles and lengths by up to 26 %, so
getting the scale right is worth more here than any amount of model tuning.

That formula assumes a square native pixel, so it is itself a hypothesis, and it was
cross-checked against the direct ruler reading on the images where both rulers are legible and
the tick value agrees. On the 800x652 group (n = 114) the rulers give a ratio of 1.005 against a
predicted 1.000; on 1200x800 from a 1196x768 frame (n = 10) they give 0.964 against 0.963. Two
independent methods agreeing to **0.5 % and 0.1 %** is why the formula is used as ground truth
below.

`roi_scale.py` reads the scanner's own ruler and never sees the masks, so the native frame size
is an independent check on it:

| check | result |
|---|---|
| Depth recovered on a frame whose scanner printed "4.5 cm" | **45.0 mm** |
| Per-image ruler read, training aponeurosis set | **93.9 %** of 1048 |
| Per-image ruler read, training fascicle set | **98.3 %** of 2761 |
| Per-image ruler read, **test set** | **80.9 %** of 309 |
| `sx/sy` vs true anisotropy, images with both rulers (n = 1263) | median relative error **0.29 %**, within 5 % for 72.8 % |

Two defects in this estimator were found and fixed, and both fixes are measured:

- **The tick is now chosen once for both rulers.** The scanner prints one scale, so a vertical
  tick of 10 mm with a horizontal tick of 2 mm in the same image is physically impossible. That
  disagreement affected 46.4 % of training images and cost a **100.6 %** median ratio error on
  them; it is now **zero images**, and the ratio error on two-ruler images is the 0.29 % above.
  Coverage was preserved exactly while doing it (see 5.4).
- **The tick candidates now include 1 mm.** With the old set, 512x512 gave a muscle thickness of
  **55.6 mm**, anatomically impossible; with 1 mm it gives **27.8 mm** and all six device
  groups land in a physiological 18 to 39 mm.

## 5. Observations

Nothing here changes section 1, and all of it was measured the same way.

### 5.1 What these numbers are and are not

- **None of this is the official metric.** That needs true PA, FL and MT, which are not
  published for the test set. MT is measured against the masks; the angle against the masks'
  dominant orientation; FL cannot be measured at all because no image carries both mask types,
  so it is only bounded.
- **Every millimetre and degree here is an optimistic upper bound** on what the test set would
  give, because 88.3 % of test images show a single ruler and fall back to isotropy.
- **The control is not a ResNet-34.** `modelos.Control` tries `segmentation_models_pytorch`
  first and silently fell back to the project's own 1.95 M U-Net, because the library is not
  installed on the training host — the parameter count gives it away against a ResNet-34's
  ~24 M. Earlier text in this file called it a "professional ResNet-34 control". The fallback
  now prints a warning and each model records which body it ran.
- **The two read-outs are not a pessimistic/optimistic pair.** Measured, they are not ordered:
  in the 1152x864 fold the free read gives 1.19 deg and the oracle 2.32 deg, so the oracle is
  *worse*. The cause is a protocol bias, not the model: the reference is the dominant
  orientation of the **whole** mask, a global quantity, and the free read also aggregates
  globally, while the oracle is restricted to the mask region and measures something more
  local. They are two different measurements, neither bounds the other, and in section 1 they
  order the architectures differently — which is why both are always reported.
- **"Lower bound on the PA error" does not hold.** PA is the difference between the fascicle
  angle and the deep aponeurosis slope, so the two errors can cancel rather than add. The claim
  is withdrawn: the orientation error is a *related* quantity, not a bound.

### 5.2 The irreducible part of the scale

With a single legible ruler the `sx/sy` ratio cannot be recovered and isotropy is assumed: on
those 2434 training images the median error is **11.1 %**. That is the floor, not a defect, and
it is why no method below that line was adopted.

The 1 mm tick correction uses the **training** masks, so it fixes the ground truth the internal
evaluation is measured against — reporting a 9.9 mm MAE against a truth that is off by a factor
of two is reporting wrongly. It is **not applicable to the test set**, where there are no masks
and the tick ambiguity cannot be broken by depth, because devices genuinely use depths from 40
to 100 mm. It does not touch the ablation: truth and prediction within a fold share the same
`sy`, so the factor scales both equally.

The evidence that the scale was at fault rather than the muscle is that **MT in pixels is
similar across every group** (68 to 103 px), so the muscle occupies a comparable fraction of
the frame everywhere:

| group | tick | MT in mm | MT in px |
|---|---|---|---|
| 660x556 | 10 | 18.4 | 75.5 |
| 1234x688 | — | 19.3 | 86.4 |
| 800x652 | 5 | 20.5 | 68.2 |
| 1152x864 | 5 | 23.2 | 103.4 |
| 800x600 | 2 | 38.6 | 93.2 |
| **512x512** | **2** | **55.6** | 84.6 |

### 5.3 Paths measured and discarded

- **Consolidating the scale by image-size group (W, H) is wrong.** It raises coverage but the
  1200x800 bucket holds 239 of the 309 test images and mixes several devices and depth settings
  (sy from 5.3 to 13.5 px/mm), so the group median is wrong for most of them.

- **Consolidating by *device fingerprint* is worse than doing nothing.** An earlier version of
  this file reported that a fingerprint (image format + ruler band + tick + span) raised test
  coverage to 98.7 %, and it does. But measured properly — holding out the entire native-frame
  group, learning without it and testing on it, which is what happens to a test image from an
  unseen device — propagating the anisotropy by fingerprint gives **26.7 %** median ratio error
  and by format **24.1 %**, against **11.1 %** for simply assuming isotropy. Filling the
  vertical scale the same way gives **69.6 %**. The version that estimates each fingerprint's
  median from the same rows it is then measured on scores **0.00 %**, and that is the trap: the
  fingerprint identifies the device so well that its median *is* that device's, and when the
  device is absent it transfers another's. Of the 25 distinct fingerprints in the test set only
  7 appear in training, and 79 of 309 test images have none. `consolidar_grupo` and
  `consolidar_huella` are kept with a do-not-wire warning and these numbers. That
  `prep_dataset.py` read the raw CSV and never called them was, by accident, the right thing.

- **Estimating the anisotropy from the resampling trace does not pay off either.** Upsampling an
  axis by `f` limits its real spectrum to `0.5/f`, so the per-axis spectral cutoff should
  recover `f` from the image alone and therefore generalise to unseen devices. The signal is
  real: a non-resampled image shows a sharp Nyquist peak at 261x the mean power while resampled
  ones show none. Calibrated with isotonic regression fitted **without the device being
  measured**, over 1434 images and 14 devices:

  | | mean | median | p90 |
  |---|---|---|---|
  | assume isotropy | **7.99 %** | 10.29 % | 20.86 % |
  | spectral cutoff, calibrated | 9.40 % | **8.59 %** | **18.21 %** |

  It wins on the median and the p90 and **loses on the mean**, which is what an MAE metric pays,
  and it wins 7 of 14 groups by mean. The pattern is not noise — it wins on every genuinely
  anisotropic group, dropping 1088x644 from 12.6 % to 2.8 % and 1152x864 from 11.1 % to 5.1 %,
  and loses only on isotropic ones where isotropy is right by definition — but the damage there
  is large: on 800x600 it injects 20.8 % where isotropy has 1.6 %. A dead zone around 1.0 makes
  it *worse* (groups won fall from 8 to 3). Using the Nyquist peak as an isotropy detector
  reaches **100 % precision** at threshold 0.42 and only **17.3 % recall**, so it would shield
  104 of 602 isotropic images and the balance stays negative. Not adopted. What is missing is
  the detector, not the estimator.

- **Twelve orientation bins are not enough.** A synthetic geometry test showed a quantisation
  error of up to 7.5 deg, above the official PA tolerance of 6 deg. Raised to 24 bins, with a
  probability-weighted circular mean interpolating inside the bin.

- **Unweighted cross-entropy never learns fascicle orientation.** In a 3-epoch control pilot the
  orientation IoU stayed at 0.000 while aponeurosis IoU reached 0.42, because background covers
  about 98 % of the pixels. Foreground class weighting was added.

### 5.4 Defects found in my own earlier work

- **The evaluation harness measured the wrong thing, and my first two diagnoses of it were also
  wrong.** A first pass in physical units reported angular errors of 83, 74 and 64 degrees by
  fold, nearly identical across all four architectures. A quantity that varies with the fold but
  not with the model cannot be measuring the model. Both wrong diagnoses are kept because the
  wrong diagnosis is part of the lesson.
  - *First hypothesis, refuted:* a 90 degree convention mismatch. Both `orientation_bins` and
    `dominant_orientation_deg` apply the `+90` that turns the structure tensor's gradient
    direction into the structure direction. Not it.
  - *Second hypothesis, also refuted, and published here before being checked:* that the harness
    integrated background bias by summing each orientation bin over the whole image. It does,
    and that is imprecise, but an independent check showed it shifts the angle by 1 to 3
    degrees, not 60 to 80.
  - *The actual cause:* `dominant_orientation_deg` weighted the structure tensor by the mask
    itself. The fascicle masks are lines of **2.00 px mean thickness**, and on a line that thin
    the Sobel gradient is ~0 *on the line*: all the energy sits in the neighbours, where the mask
    is 0. Measured on lines with a known answer: mean error **54.7 deg** with the weight (a
    horizontal line returned -90 deg, the largest error possible) against **1.6 deg** without
    it. The broken quantity was the *ground truth*, not the prediction. With a correct reference
    the same fold goes from 82.6 deg to **1.19 deg**, confirmed end to end by this harness and
    independently at 1.2 deg by a reviewer's own code.
  The fix is one line, and `test_geometria.py` now carries a regression test on 1 px lines — the
  original constructed case used thick strokes, which is why no test caught it.

- **A NaN in the truth silenced four folds out of five.** The harness checked that the *model's*
  angle was finite but not the truth's. A fascicle image without a physical scale gives `NaN`,
  it entered the list, and one value left the whole mean at `NaN`. In the 1152x864 fold **2
  images out of 426** were enough. Fascicle coverage by fold is 99.5, 83.5, 87.0, 92.3 and
  100 %, and the only fold that ever reported a number was the only one at 100 %. Images without
  a scale are now dropped **and counted** (`n_fasc_sin_escala`), and every mean goes through a
  helper that says so out loud instead of returning NaN silently.

- **The first version of the common-tick fix cost coverage.** Forcing one tick value discarded
  the whole ruler when either reading fell outside its physiological range, dropping test
  coverage from 80.9 % to **55.3 %**. Each ruler is now retried with its own value and only the
  *ratio* is discarded (isotropy) while the *scale* is kept. One gives up what cannot be known,
  not what has already been measured.

- **An implementation error in the spectral estimator.** Measuring one axis's cutoff over 23
  samples and the other's over 522 is not a comparison. With a **square** window the correlation
  doubled, from +0.25 to +0.50. The conclusion in 5.3 is the one measured after this fix.

- **The epoch used to be picked on the fold being reported.** The swing between epochs within
  one arm reaches 0.24 to 0.61 while the differences being measured are ±0.017 — an order of
  magnitude smaller, so the selection was picking noise. No figure from that scheme is used
  anywhere in this file.

- **Linearised error propagation for FL is not a mean error.** Propagating
  `FL = MT/sin(PA)` as `dFL = dMT/sin(PA) + MT*cos(PA)/sin^2(PA)*dPA` sums magnitudes although
  the derivative is negative, so it is a worst-case bound and was mislabelled as a propagated
  MAE. Fed with the 80 deg errors above it returned FL of 300 to 600 mm, outside the
  physiological 30 to 200 mm range. It is now simulated over the physiological PA range and
  declared non-informative instead of printing an absurd number.
