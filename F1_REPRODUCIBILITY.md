# F1 ensemble: reproducibility and competition evidence

**Competition:** [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
**Team:** Francisco Angulo de Lafuente (Agnuxo1)
**Date:** 31 October 2026
**Best public Kaggle score:** **0.23862** (submission **57852101**, provisional public leaderboard position **29**).
**Winner eligibility:** NOT YET CERTIFIED. Full end-to-end reproducibility on new images remains a mandatory separate check.

## What F1 actually does

F1 is a **fixed-weight average of three image-derived prediction families**:

```text
Own family = (U04 + U07) / 2
F1 = (Own family + Honghanhhh + Soukeaizenz) / 3
```

This formula is applied separately to **pennation angle**, **fascicle length**, and
**muscle thickness**. There is no per-image tuning, hidden-label training, or leaderboard-fitted
coefficient. The original input model families must themselves be runnable on new images.

- **U04:** Our inner-aponeurosis-border fascicle length adaptation of the public
  `watehell/ans-is-simple-physical-scale-u-net-geometry` approach. Source currently archived
  under the original UMUD project sprint's `codigo/k_u04/`.
- **U07:** Our own UNet++/EfficientNet-B3 trained on the competition's disclosed
  aponeurosis and fascicle masks, followed by U04's geometrical measurement stage. Source
  currently archived under `codigo/k_u07/`.
- **Honghanhhh:** Public Kaggle `umud-pipeline-v2-lb-0-3389` model family.
- **Soukeaizenz:** Public Kaggle `anatomynet-biomechanical-u-net-0-399-1-sota` model family.

The corresponding public notebooks must be properly attributed, license-checked,
and made available in an end-to-end re-execution plan. The Kaggle host specifically requires
that a winning ensemble be reproducible on a new dataset without a leaderboard.

## Regenerate the ensemble from independently produced model predictions

First run each licensed model inference pipeline on the same ordered set of new images,
using `image_id,pa_deg,fl_mm,mt_mm` as the output columns. Then run:

```bash
python ensemble_f1.py \
  --u04 u04_predictions.csv \
  --u07 u07_predictions.csv \
  --honghanhhh honghanhhh_predictions.csv \
  --soukeaizenz soukeaizenz_predictions.csv \
  --output f1_submission.csv
```

The script checks that the IDs match exactly and rejects missing or non-finite values.
No competition images, trained masks, or raw predictions are distributed in this repository.

## Independent exactness check

On the **309 original test image IDs**, `ensemble_f1.py` reconstructed the F1 candidate
recorded before today's submissions, with maximum absolute discrepancies of **0.001**
per variable due solely to CSV rounding. The recomputation was checked against
the frozen `F1_W47_hh_sa.csv` source (309 rows).

Local audit evidence is at `E:/UMUD-work/submission_audit/` on the research machine.
The audited official submission artifacts and their SHA-256 hashes remain private,
consistent with the competition's data redistribution restrictions.

## All official exploratory submissions (31 October 2026)

| Submission ID | Method | Public UMUD Score | Decision |
|---|---|---:|---|
| 57852073 | Standalone U-Net + NeuroPixel hybrid (diagnostic) | 4.44858 | Rejected: unstable physical geometry, especially fascicle length saturation |
| **57852101** | **F1 fixed three-family average** | **0.23862** | **Best public candidate to date** |
| 57852108 | F3 four-family blend | 0.24554 | Rejected |
| 57852110 | F4 robust five-model median | 0.25845 | Rejected |
| 57852147 | F1 + image-similarity temporal stabilization | 0.23997 | Rejected: does not improve F1 |

The prior best was **0.24237** (submission **57799195**). The F1 improvement
is approximately **1.55%** relative on the public metric. The leader's public score
on 31 October was **0.18297**. Only ~33% of images contribute to the public leaderboard;
the final ranking uses the remaining ~67%, so public improvements are not guarantees.

## Remaining work before the final awards

1. Prepare truly executable, licensed versions of **every component model** for fresh data:
   include package versions, preprocessing, checkpoint acquisition/training and inference.
   A CSV-combination script alone **does not** satisfy the organizer's reproducibility requirement.
2. Confirm all third-party notebook licenses and attributions; do not redistribute the
   competition images or masks, and do not expose private participant data.
3. Repeat group- and device-disjoint validation, checking duplicate images across folds.
   The official organizers warn that training duplicates may inflate local validation.
4. Correct the standalone hybrid's severe fascicle-length error **before** reusing it.
5. Before the **14 November 2026, 23:00 UTC** close, select the best two final Kaggle
   entries based on valid competition evidence. Do not select failed diagnostic submissions.

## Interpretation and limitation

The reported 0.23862 is a **provisional public leaderboard score**, not the private score
and not a claim to be eligible for a prize. The published standalone NeuroPixel method did
not produce that score. A genuinely runnable ensemble remains to be packaged before
the winner's open-source and FAIR evaluation.
