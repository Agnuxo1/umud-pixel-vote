# umud-pixel-vote — muscle architecture from ultrasound by per-pixel voting

Automatic estimation of **pennation angle (PA)**, **fascicle length (FL)** and **muscle
thickness (MT)** from single B-mode ultrasound frames, for the
[UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
(Swissuniversities, Kaggle, closes 2026-11-14).

Author: Francisco Angulo de Lafuente ([Agnuxo1](https://github.com/Agnuxo1)).
License: **GPL-3.0** (the licence the competition requires of winning submissions).

## The problem, and why it is not just segmentation

The challenge publishes **no PA/FL/MT labels at all**: the training set contains 1048
aponeurosis masks and 2761 fascicle masks, and **no image carries both kinds**. The targets
are in millimetres and degrees, while the released images have been **resized anisotropically**
(most to 1200x800) with the masks kept at the native frame size. So before any geometry is
possible, the **physical scale of each axis has to be recovered from the image itself**.

## What this repository contains

| module | role |
|---|---|
| `roi_scale.py` | physical scale (px/mm per axis) **without OCR**: finds the scanner's ruler by the regularity of its tick train (multiscale band sweep with a sliding maximum), rejects text columns by peak width, ink fraction and brightness uniformity, and picks the tick value (2/5/10 mm) by physiological range |
| `geometria.py` | the challenge's own annotation procedure, in code: MT as the perpendicular distance between the two aponeuroses at three sites, PA as the fascicle angle relative to the **deep** aponeurosis in metric space, FL = MT / sin(PA) (the linear extrapolation the annotators use) |
| `modelos.py` | three bodies with one identical head: a professional U-Net control, a capacity-matched small control, and `NeuroPixelField` |
| `train.py` | one vocabulary, partial supervision per image, validation grouped by device fingerprint |
| `infer.py` | votes to millimetres and degrees; the geometry stage is identical for every model, so the ablation isolates the contribution of the own layer |

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

Here the dictionary is `{background, superficial aponeurosis, deep aponeurosis}` plus twelve
fascicle-orientation bins. `NeuroPixelField` has **58 096** parameters against **1 949 583** for
the professional U-Net control; parameter counts are reported next to every metric.

## Honesty about the leaderboard

The public leaderboard is computed on roughly 102 images and has received over 15 000
submissions, while the private test set contains an ultrasound device and a patient population
absent from training. Nothing here is calibrated against the public leaderboard, and the
organizers' own final criterion — whether the submission can be re-run on a new dataset where
no leaderboard exists — is treated as binding.

## Results

Reported per parameter, with seeds and variance, as experiments complete. Negative results are
kept. See `RESULTS.md`.

## Reproducing

The competition data is distributed under CC BY-NC-SA and **may not be redistributed**, so it
is not included here. Download it from the competition page, then:

```bash
pip install -r requirements.txt
python prep_dataset.py          # compact 384x256 working set + per-image physical scale
python run_all.py               # 4 architectures x 3 seeds
python infer.py --ckpt runs/neuropixel_s0.pt --split test --out submission.csv
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

## Configuration

Data locations are read from the environment, so nothing is hard-coded to one machine:

```bash
export UMUD_RAW=/path/to/competition/data      # the unzipped competition folder
export UMUD_WORK=/path/to/scratch              # where the working set and CSVs are written
```
