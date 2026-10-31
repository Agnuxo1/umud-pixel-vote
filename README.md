# umud-pixel-vote — muscle architecture from ultrasound by per-pixel voting

Automatic estimation of **pennation angle (PA)**, **fascicle length (FL)** and **muscle
thickness (MT)** from single B-mode ultrasound frames, for the
[UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
(Swissuniversities, Kaggle, closes 2026-11-14).

Author: Francisco Angulo de Lafuente ([Agnuxo1](https://github.com/Agnuxo1)).
License: **GPL-3.0** (the licence the competition requires of winning submissions).

## Results

20 runs completed: 4 architectures x 5 leave-one-device-out folds, paired by fold, with the
epoch picked on a **second** held-out device so the reported group is never seen during
training.

- **The NeuroPixel layer does help.** The hybrid beats the U-Net control on MT in **4 of 5**
  folds (4.63 mm against 4.90 mm on average) and on the mask-referenced angle in **4 of 5**, and
  it is the best of the four arms on three of the four summary measures. Added through a
  zero-initialised gate, at step 0 it *is* the control, so the difference is attributable to the
  layer alone.
- **The margin survives weighting.** The folds are device groups ranging from 13 to 467
  images. Weighted by images instead of by fold the MT margin **widens**, 0.27 mm to 0.41 mm,
  because the hybrid gains most on the largest group; dropping every fold under 50 images
  still leaves 3 of 4 wins.
- **NeuroPixel alone localises badly and orients well**: worst of the four on MT, but on the
  angle measured inside the true mask it beats its **capacity-matched** control in 3 of 5 folds
  with a paired median of −0.40 deg, using 58 k parameters against that control's 123 k. It
  does *not* beat the large arms once they are paired by fold — see the correction in
  [`RESULTS.md`](RESULTS.md) §1.
- **The scale is recovered from the scanner's own ruler, without OCR**, on 93.9 % of training
  and 80.9 % of test images, with a 0.29 % median ratio error where both rulers are legible.
- **The geometry stage is verified against a constructed case** with a known answer: MT exact to
  20.00 mm, the orientation of a 1 px line to within 0.1 deg.

The finding worth reading first: **the training loss ordered the arms backwards.** By masked
cross-entropy the hybrid looked worse in 4 of 5 folds; in millimetres and degrees — what the
competition pays — it is better in 4 of 5. Selecting and concluding by the loss, the normal
thing to do, would have published the opposite verdict.

Full numbers, the protocol, and the limits of each figure: [`RESULTS.md`](RESULTS.md).

## The problem, and why it is not just segmentation

The challenge publishes **no PA/FL/MT labels at all**: the training set contains 1048
aponeurosis masks and 2761 fascicle masks, and **no image carries both kinds**. The targets
are in millimetres and degrees, while the released images have been **resized anisotropically**
(most to 1200x800) with the masks kept at the native frame size. So before any geometry is
possible, the **physical scale of each axis has to be recovered from the image itself**.

## What this repository contains

| module | role |
|---|---|
| `roi_scale.py` | physical scale (px/mm per axis) **without OCR**: finds the scanner's ruler by the regularity of its tick train (multiscale band sweep with a sliding maximum), rejects text columns by peak width, ink fraction and brightness uniformity, and picks the tick value (1/2/5/10 mm) **once for both rulers**, since the scanner prints a single scale |
| `geometria.py` | the challenge's own annotation procedure, in code: MT as the perpendicular distance between the two aponeuroses at three sites, PA as the fascicle angle relative to the **deep** aponeurosis in metric space, FL = MT / sin(PA) (the linear extrapolation the annotators use) |
| `modelos.py` | four bodies with one identical head: a 1.95 M U-Net control (an smp ResNet-34 is tried first and the fallback now says out loud when it is not available), a capacity-matched 123 k small control, `NeuroPixelField` (58 k), and the hybrid |
| `train.py` | one vocabulary, partial supervision per image, leave-**two**-devices-out: one group picks the epoch, another is reported and is never evaluated during training |
| `infer.py` | votes to millimetres and degrees; the geometry stage is identical for every model, so the ablation isolates the contribution of the own layer |
| `ponderar_por_n.py` | rereads the ablation weighting each fold by its image count, and reports every comparison paired by fold — the folds range from 13 to 467 images, so a mean over folds and a mean over images are not the same number |
| `run_semillas.py` | seed spread **inside** a fixed fold: `train.py` derives the fold from the seed, so the published run has one seed per fold and no measure of initialisation noise. This pins `--val-group`, moves only the seed, and refuses to report if the cells do not share a fold |

## The hypothesis under test: NeuroPixel as a per-pixel voting field

[NeuroPixel](https://github.com/Agnuxo1) keeps the whole neural state on a spatial canvas.
`NeuroPixelField` carries its three defining pieces over to this problem:

1. a **retina** of dilated convolutions (rates 1, 2, 4, 8; receptive field about 31 px) that
   gives every pixel an identity vector **without ever reducing resolution**;
2. a **neural cellular automaton** — depthwise 3x3 perception plus a 1x1 residual MLP with a
   stochastic fire rate — so line structure propagates by iteration between neighbours instead
   of by down- and up-sampling;
3. a **tied-dictionary read applied to every pixel**, which is literally each pixel casting a
   vote for one word of a shared dictionary.

Here the dictionary is `{background, superficial aponeurosis, deep aponeurosis}` plus
twenty-four fascicle-orientation bins, with a probability-weighted circular mean interpolating
inside the bin. `NeuroPixelField` has **58 288** parameters against **1 949 979** for the U-Net
control that actually ran; parameter counts are reported next to every metric, and the tables
state which body each arm used.

## Reproducing

The competition data is distributed under CC BY-NC-SA and **may not be redistributed**, so it
is not included here. Download it from the competition page, then:

```bash
pip install -r requirements.txt
python prep_dataset.py          # compact 384x256 working set + per-image physical scale
python run_all.py               # 4 architectures x 5 device folds
python infer.py --ckpt runs/neuropixel_s0.pt --split test --out submission.csv
```

Data locations are read from the environment, so nothing is hard-coded to one machine:

```bash
export UMUD_RAW=/path/to/competition/data      # the unzipped competition folder
export UMUD_WORK=/path/to/scratch              # where the working set and CSVs are written
```

## Attribution

- Official metric and the `DL_Track_US 0.3.1` aponeurosis notebook: the competition organizers
  (Paul Ritsche et al.).
- Domain algorithms cited as prior art: [DL_Track_US](https://github.com/PaulRitsche/DL_Track_US),
  [UltraTimTrack](https://github.com/timvanderzee/UltraTimTrack),
  [DUSTrack](https://github.com/praneethnamburi/DUSTrack).
- Public Kaggle notebook `watehell/ans-is-simple-physical-scale-u-net-geometry` established that
  per-axis physical scale is the dominant factor and recovers it by OCR of the depth label plus
  ruler ticks. The estimator here is an independent, OCR-free reimplementation with explicit
  physiological validation; the competition rules allow reuse of existing code in an improved
  version, and the idea is credited here.

## Observations

**On the leaderboard.** The public leaderboard is computed on roughly 102 images and has
received over 15 000 submissions, while the private test set contains an ultrasound device and
a patient population absent from training. Nothing here is calibrated against the public
leaderboard, and the organizers' own final criterion — whether the submission can be re-run on
a new dataset where no leaderboard exists — is treated as binding.

**On the figures above.** None of them is the official metric, which needs labels the
competition does not publish for the test set; MT is measured against the masks and the angle
against the masks' dominant orientation. Because 88.3 % of test images show a single legible
ruler and fall back to isotropy, every millimetre and degree here is an optimistic upper bound
on what the test set would give.

**On the loss.** The quantity the training minimises ranks the arms the other way round: by
masked cross-entropy the hybrid is worse than the control in 4 of 5 folds and NeuroPixel alone
is worse than all three others in 5 of 5. Both readings are published side by side in
`RESULTS.md`; the disagreement between them is the finding, and nothing here claims the loss
agrees.

**On the audit.** An earlier version of these documents was refuted by an independent review,
which is how three of the recorded defects were found. The present version has **not yet been
independently audited**; an audit is scheduled before the competition closes.

**On negative results.** They are kept rather than dropped, including the ones that refuted
earlier claims of my own: four statements previously published in this repository were wrong
and are recorded as such, next to the measurements that refuted them — three found by the
independent review, the fourth found here by pairing a mean I had reported unpaired. Section 5
of [`RESULTS.md`](RESULTS.md) collects them, together with the three paths that were measured
and discarded for being no better than the trivial alternative.

**On what is still unmeasured.** There is no measure of seed spread in this repository. Because
`train.py` derives the reported fold from the seed, the five "seeds" of the published run are
the five device folds, one seed each, and the margins above are therefore not known to exceed
initialisation noise. `run_semillas.py` is the experiment that answers it and is queued on a
private GPU server behind other work; if it has not run by the time the audit does, the
statement stands as it is written here.
