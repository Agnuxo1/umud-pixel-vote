# Results

**Headline.** `NeuroPixelField` is a **specialised orientation head**, and on one ultrasound
device it is very good at that job: with the fascicle mask given, it measures fascicle
orientation to **1.42 ± 0.23 deg** against the U-Net control's **4.58 ± 0.26** and the
capacity-matched control's **3.06 ± 0.94** — three times better than a control with **33 times
more parameters**, over 3 seeds, with a margin thirteen times the seed-to-seed spread. On a
different device it does not win at all, and on muscle thickness it is the worst of four arms
in every fold: it knows which way a fascicle runs, not where it is. That split — excellent
orientation, poor localisation, and both device-dependent — is the finding, and it is a
different capability from the one this competition ranks.

Around it, a pipeline that stands on its own: the organizers' metric reproduced, the **physical
scale recovered from the scanner's own ruler without OCR** on 93.9 % of training and 80.9 % of
test images with a 0.29 % median ratio error, and a geometry stage verified against a
constructed case to **0.1 deg**.

And two methodological findings that outlast the ranking. First, **the training loss orders the
architectures backwards** with respect to the millimetres and degrees the competition pays.
Second, and this one cost me the previous version of this paragraph: **the ablation margin does
not survive its own seed noise.** Measuring seed spread inside a fixed fold — the control this
study was missing — the hybrid's advantage on muscle thickness lands *inside* the noise in both
folds measured, and in one of them the sign flips. §1 reports that in full. An ablation that
dismantles its own headline when the missing control is added is worth more than the headline
was.

Everything here was measured on this machine and on a private RTX 3090 server; nothing was
calibrated against the Kaggle public leaderboard. Section 5, **Observations**, carries the
limits of these numbers, the paths that were measured and discarded, and the defects found in
my own earlier work — nothing is omitted, it is simply not what the result is.

**Audit status.** An earlier version of this file was refuted by an independent review, which is
how three of the defects in 5.4 were found. The present version has **not yet been through an
independent audit**; one is scheduled before the competition closes.

## 1. What the NeuroPixel layer contributes

Four bodies — U-Net control **1 949 979** params, capacity-matched small control **122 907**,
`NeuroPixelField` **58 288**, hybrid **2 009 752** — trained on 5 leave-one-device-out folds,
20 runs in total. The hybrid mixes the NeuroPixel layer into the control through a
zero-initialised 1x1 convolution, so **at step 0 it is exactly the control**: any difference
is attributable to the layer and to nothing else.

Selection is separated from reporting: a **second** device group is held out to pick the
epoch, the epoch is chosen at the minimum of a 3-epoch moving average, and the reported group
is **never evaluated during training**.

MT error in millimetres, by reported fold (lower is better). The folds are device groups and
are nowhere near the same size, so the count is part of the table:

| fold | n images | control | control_small | hybrid | neuropixel |
|---|---|---|---|---|---|
| 1152x864 | 467 | 7.03 | 7.97 | **6.26** | 16.34 |
| 512x512 | 139 | 4.97 | 5.64 | **4.71** | 11.65 |
| 660x556 | 267 | 3.76 | 4.18 | **3.65** | 12.79 |
| 800x600 | 13 | 1.31 | 1.10 | **0.49** | 14.32 |
| 800x652 | 72 | **7.44** | 7.68 | 8.04 | 14.63 |
| mean of folds | | 4.90 | 5.31 | **4.63** | 13.95 |
| weighted by n | 958 | 5.77 | 6.46 | **5.36** | 14.51 |

Orientation error in degrees, both read-outs (fascicle-mask counts per fold: 424, 66, 147, 72,
43; 752 in all):

| | control | control_small | hybrid | neuropixel |
|---|---|---|---|---|
| free read (no label), mean of folds | 3.66 | 3.45 | **3.36** | 5.04 |
| free read, weighted by n | 2.61 | **2.44** | **2.44** | 3.27 |
| oracle read (uses the label), mean of folds | 2.49 | 2.40 | 2.23 | **1.78** |
| oracle read, weighted by n | 2.54 | 2.15 | 2.22 | **2.08** |
| metric-shaped proxy, oracle | 1.72 | 1.74 | **1.54** | 3.37 |

### 1.1 The one result that survives every control: a specialised orientation head

On the angle measured **inside the true mask**, on the 512x512 device, over **3 seeds**:

| arm | params | oracle angle (deg) | seed spread |
|---|---|---|---|
| `NeuroPixelField` | **58 288** | **1.42** | ± 0.23 |
| capacity-matched control | 122 907 | 3.06 | ± 0.94 |
| U-Net control | 1 949 979 | 4.58 | ± 0.26 |
| hybrid | 2 009 752 | 3.94 | ± 0.89 |

Three times better than a control with 33 times the parameters, and the margin is **thirteen
times the seed-to-seed spread** — this is not a one-seed accident. On the 1152x864 device the
same arm does **not** win (2.62 ± 0.83 against the small control's 1.94 ± 0.10), and on muscle
thickness it is the worst of the four in all five folds, by +9.0 mm of median difference: it
does not find the aponeuroses at all.

So the capability is real, measured, cheap, and **device-dependent**: an orientation head, not a
muscle-architecture model. Nothing here claims it generalises to devices it has not seen — the
opposite is what the two folds show.

### 1.2 The ablation margin does not survive its own seed noise

This is the control the study was missing, and the reason the previous headline is gone.
`train.py` derives the reported fold from the seed, so the published 20-run pass has **one seed
per fold** and no measure of initialisation noise. `run_semillas.py` pins the fold and moves
only the seed; `dispersion_semilla.py` compares the margin against the spread. Two of the five
folds finished. The other three never ran: the server rejected them with
`rejected:project_expired` because this project's queue window had lapsed, and a new window is
not mine to grant — so unless one is assigned, the seed measurement stands at **2 of 5 folds**,
with 3 seeds per cell:

| fold | n | hybrid − control (MT, mm) | pooled SD | std. error | ratio |
|---|---|---|---|---|---|
| 1152x864 | 467 | −0.518 | 0.774 | 0.632 | **0.82** |
| 512x512 | 139 | **+0.616** | 1.136 | 0.927 | **0.66** |

Neither reaches 2, so neither is distinguishable from noise. And in 512x512 **the sign flips**:
with the single published seed the hybrid won by −0.261 mm; over three seeds it loses by
+0.616 mm. Its three values in that fold are 7.72 / 5.66 / 4.71 mm — a 3.01 mm range between
seeds against a published margin of 0.26 mm, **eleven times larger**. On the angles the ratio
runs 0.91 to 1.50, never above 2 either.

**What this retracts.** The claim that the NeuroPixel layer improves muscle thickness when mixed
into the control. The paired counts (4 of 5 folds), the weighting check and the
drop-the-small-fold check were all computed correctly and all reported one seed per cell, so
they measured a quantity whose own noise is larger than the effect. Keeping the counts and
dropping the seed control would be the easy version of this paragraph; the counts are still in
§5.1 with their figures, and this is what they are worth.

**What it does not touch.** The physical scale (§4), the geometry stage (§3), the robustness of
NeuroPixel's orientation win in 512x512 and of its localisation failure — all of those have
margins several times their seed spread — and the loss finding below, which is about the
*ordering* of two read-outs and not about any margin.

**The training loss ordered the arms backwards.** By masked cross-entropy on the selection
group the hybrid was *worse* than the control in 4 of 5 folds. In millimetres and degrees it was
better in 4 of 5. Selecting and concluding by the loss — the normal thing to do — would have
published the opposite ordering of the arms. That disagreement is the finding, and it stands
whether or not either margin clears its noise: two quantities that are supposed to track each
other rank four architectures in opposite orders.

For the record, the earlier correction still applies and is now doubly true: "best mean of the
four arms" on the oracle angle was a reading of a column of means without pairing, and the
0.45 deg it showed came from one device group. §5.4 keeps it as a defect of mine.

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

- **Seed spread is now measured in 2 of the 5 folds, and it is larger than the margin.** See
  §1.2: the paired counts below (4 of 5 folds on MT, and the weighting and drop-the-small-fold
  checks) are each computed from **one seed per cell**, because `train.py` picks the reported
  fold from the seed (`vg = usables[seed % 5]`). They are kept here with their figures, and §1.2
  states what they are worth: in both folds where seeds were added the hybrid–control difference
  sits inside its own noise, and in one the sign reverses. The remaining three folds are
  re-queued. Read any count in this subsection as a one-seed count.
- **One fold carries 13 images.** The device groups range from 13 to 467 images with an
  aponeurosis mask (467, 139, 267, 13, 72) and from 43 to 424 with a fascicle mask. A mean over
  folds is not a mean over images, and the two checks this invites are reported in §1 rather
  than here because the MT result passes both.
- **The selection loss says the opposite, and here it is in full.** Masked cross-entropy on the
  selection group, which is the quantity the training actually minimises, ranks the arms the
  other way round: control **0.2267**, control_small 0.2500, hybrid 0.2649, neuropixel 0.3808
  (means over folds). Paired by fold, the hybrid is worse than the control in **4 of 5** and
  NeuroPixel alone is worse than all three others in **5 of 5**, by a wide margin (+0.1541
  against the control, +0.1308 against the capacity-matched small control, so capacity does not
  explain it). Section 1 reports the physical units because that is what the competition scores,
  and the disagreement between the two readings is itself the finding — but both are on the
  table, and nothing here claims the loss agrees.
- **The advantage is clearest on MT and on the mask-referenced angle**, 4 of 5 folds each. On the
  free angle read, which uses no label, the hybrid wins only 3 of 5, and on the metric-shaped
  proxy 3 of 5 against the control. The result should not be read as a general improvement
  across all three target quantities: FL is not measurable here at all.
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

### 5.2 The irreducible part of the scale — and the test set confirmed it

With a single legible ruler the `sx/sy` ratio cannot be recovered and isotropy is assumed: on
those 2434 training images the median error is **11.1 %**. That is the floor, not a defect, and
it is why no method below that line was adopted.

**The stated upper-bound caveat was right, but it is not the main cause of the test score, and
I said it was.** This pipeline scored **4.44858** against 0.23862 for an average of public
notebooks on the same account. My first explanation here blamed the scale estimator's coverage.
It cannot be: an 11.1 % ratio error yields a score of **0.709**, and 4.45 needs roughly a 70 %
relative error. The measured breakdown is in §6; scale accounts for the muscle-thickness part of
it, about 23 %, because the 1 mm tick correction uses training masks and is **not applicable to
the test set**, where absolute scale therefore goes uncorrected. The other 77 % is the
orientation head failing to transfer. The caveat was a real limit, honestly stated, and I then
used it to explain a number it does not explain.

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

- **I published an ablation verdict without the seed control, and it did not survive it.** Every
  version of this file until now headlined that the NeuroPixel layer improves muscle thickness
  in 4 of 5 folds. The counts were right; the design was not, because `train.py` ties the fold
  to the seed and so the pass had one seed per cell and no way to see initialisation noise. With
  3 seeds in 2 folds the margin is inside the noise and in one fold the sign flips (§1.2). The
  mitigating detail is that this was a stated open limitation and the experiment to settle it was
  already queued with the outcome committed to in advance; the damning one is that the headline
  was published anyway, for three days, while the control was missing.
- **I published a mean as a win without pairing it, which is the mistake the audit had already
  caught me making twice.** An earlier version of §1 said NeuroPixel alone has "the best mean of
  the four arms" on the oracle angle, 1.78 deg against the hybrid's 2.23. The column is right
  and the reading was wrong: paired by fold, that 0.45 deg comes from a single device group
  (512x512, −2.53 deg), and across the other four NeuroPixel alone is behind in three, paired
  median **+0.27 deg** against the hybrid. I found this one myself, by weighting the folds by
  their image counts (`ponderar_por_n.py`) because the groups turned out to range from 13 to 467
  images. What the check does *not* do is damage the main result: the hybrid's MT margin widens
  under the same weighting, from 0.27 to 0.41 mm.
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

## 6. The test score, diagnosed

This pipeline was submitted once and scored **4.44858** on the public leaderboard, against
**0.23862** for an average of public notebooks on the same account. This section is the
measurement of why, because a number like that with no explanation is not a result — and because
my first explanation of it, blaming the scale estimator, was wrong by arithmetic: an 11.1 % ratio
error gives 0.709, while 4.45 needs roughly a 70 % relative error.

Breakdown, using the public ensemble as a rough stand-in for the hidden truth:

| quantity | this pipeline | ensemble | contribution to the score |
|---|---|---|---|
| `fl_mm` | 200.00 | 87.09 | **9.41** of 13.35 — 70 % |
| `mt_mm` | 30.22 | 20.83 | 3.13 — 23 % |
| `pa_deg` | 8.17 | 15.92 | 1.29 — 10 % |

**Fascicle length dominates, and its value is exactly the 200 mm clip ceiling.** On the unseen
test devices the orientation head predicts **horizontal with high confidence**: median fascicle
angle **0.44 deg** and circular concentration **R = 0.99**, so the model is not uncertain, it is
confidently wrong. The deep aponeurosis is recovered correctly (median slope −0.65 deg), so
`PA = |a_f − a_d|` lands near 1 deg, the `pa_lo = 5` clip lifts it, and `FL = MT / sin(5 deg)`
= 11.5 × MT hits the ceiling in **18 of 30** images.

This is §1.1 taken to its limit rather than a new defect: an orientation head that wins on one
device (1.42 ± 0.23 deg) and does not win on another does not transfer to scanners it has never
seen. The capability is real and device-bound, and the test set is built from devices absent from
training.

Two hypotheses tested and **refuted** on the way, both kept because the wrong ones are part of
the measurement:

- *That the read-out was at fault* — that `infer.medir` averages orientation over the whole
  inter-aponeurosis region instead of weighting by predicted fascicle mass. Implemented the
  alternative: PA 8.73 against 8.34 deg. Identical. Not the read-out.
- *That the scale was at fault* — ruled out as the dominant cause by the arithmetic above. It
  does explain the muscle-thickness share: the 1 mm tick correction uses training masks and is
  not applicable to the test set, so absolute scale goes uncorrected there.

### 6.1 The clip artefact was 39 % of the score, measured by submitting

Nothing about the model can be fixed without retraining, but **publishing the ceiling of a clip
is not a prediction**. When PA lands on `pa_lo` the estimator has no measurement of PA, and
dividing by `sin(PA)` turns that absence into 200 mm. `infer.medir` already carries a
physiological fallback for unresolved geometry; `envio_sin_saturar.py` uses it for this case too
and changes nothing else — no tuned threshold ("PA hit the clip" is a binary condition of the
formula) and MT untouched, being a direct measurement of the two curves.

Submitted. **4.44858 → 2.69767**, a drop of **1.751, or 39.4 %**. So that much of the own
pipeline's score was the clip artefact and the remaining 2.70 is the model genuinely failing to
transfer. Of 309 test images, 184 had FL pinned at the ceiling before and none after.

Two things worth stating about this number:

- **Only 56 of 309 images (18 %) yield a real measurement.** 61 do not resolve the geometry at
  all and 192 land in a clip, so the submission is mostly the fallback. That is what the pipeline
  can honestly say about this test set, and it is the more informative version of "4.45".
- **My own forecast was wrong by a point.** Using the public ensemble as a stand-in for the hidden
  truth predicted ~1.73; the real value is 2.70. The ensemble is a decent reference for ranking
  the error terms — it did identify FL correctly — but it is not a proxy for the truth, and
  anything estimated against it should be read as a lower bound on the error.
