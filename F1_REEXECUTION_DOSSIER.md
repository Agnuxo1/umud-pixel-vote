# F1 ensemble: re-execution dossier

**Competition:** [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
**Author:** Francisco Angulo de Lafuente (Agnuxo1)
**Written:** 2 November 2026. Everything stated here as verified was checked on that date.

## Why this document exists

The host added a second official criterion, and pointed it expressly at ensembles:

> can your final submission be (re-)run on a new dataset where no leaderboard exists?

Our best public entry (**0.23862**, submission 57852101, rank 50 of 606 as verified on 7 November
2026) is an ensemble. A script
that averages four CSV files does not answer that question: what has to be re-runnable is every
component that produces those CSVs. This dossier states, component by component, what exists, what
its licence is, and what it would take to run it on new images. Where something is not resolved, it
says so instead of implying otherwise.

## The four components, with licences verified on 2 November 2026

F1 is a fixed-weight average, applied separately to PA, FL and MT:

```text
Own family = (U04 + U07) / 2
F1 = (Own family + Honghanhhh + Soukeaizenz) / 3
```

| Component | Source | Licence | External dependencies | Re-runnable on new images |
|---|---|---|---|---|
| **U04** | ours, `franciscoangulo/umud-u04-inner-fl-private`; inner-aponeurosis-border FL adaptation of the public watehell approach | our code; **loads third-party weights, see below** | `watehell/umud-r34-unet-weights` (two checkpoints), plus an unpinned `pip install segmentation-models-pytorch rapidocr-onnxruntime` — so it also pulls an **OCR** package at run time | yes, but only with those weights or with retrained substitutes, and only if the unpinned packages still resolve |
| **U07** | ours, `franciscoangulo/umud-u07-own-seg-private`; UNet++/EfficientNet-B3 on the released masks, then U04's geometry stage | ours | no Kaggle dataset (`dataset_sources: []`), but it **downloads ImageNet encoder weights** at build time (`encoder_weights="imagenet"`) | yes, from the competition data plus those public pretrained weights |
| **Honghanhhh** | public notebook `honghanhhh/umud-pipeline-v2-lb-0-3389` | **Apache 2.0** | none; no GPU required | yes |
| **Soukeaizenz** | public notebook `soukeaizenz/anatomynet-biomechanical-u-net-0-399-1-sota` | **Apache 2.0** | none; GPU required | yes |

How each licence was established: the Kaggle API exposes no licence field for notebooks, so each
one was read from its public notebook page. The two third-party notebooks and the watehell notebook
all carry **Apache 2.0**, which is compatible with the GPL-3.0 selected for this repository; the contest requires an OSI-compatible open-source licence, not GPL-3.0 specifically. What is
**still outstanding**: confirming that the licence shown today is the licence of the notebook
*version* these predictions came from.

**Corrected on 7 November 2026, after an independent audit.** The dependency column above was
written from Kaggle metadata alone, which records attached datasets but not artifacts fetched over
the network, so the first version of this table said U07 had no external dependencies. Reading the
component code shows two it missed: U07 builds its encoder with `encoder_weights="imagenet"`, and
U04 installs `segmentation-models-pytorch` and `rapidocr-onnxruntime` without pinning a version,
the second of which is an OCR engine. Neither is a licence problem — both are permissively licensed
public artifacts — but both are things a re-execution has to obtain, and an unpinned install is not
reproducible as written. Note also that the own pipeline's claim of recovering physical scale
**without OCR** is about `roi_scale.py` in this repository; the U04 component of this ensemble does
use OCR.

**Where the component code lives.** `umud_u04.ipynb` and `umud_u07.py` are **not in this
repository**: they sit in a working folder from the sprint that produced them. The combination
script `ensemble_f1.py` is here, but a third party cannot today re-run the two own components from
this repository alone. That is a gap in this dossier's own promise and it is stated rather than
implied.

## The one unresolved dependency, and how to remove it

U04 loads two checkpoints from the Kaggle dataset
[`watehell/umud-r34-unet-weights`](https://www.kaggle.com/datasets/watehell/umud-r34-unet-weights):
`apo_resnet34_384x576_f0.pt` (aponeurosis, hold-out Dice 0.82) and `fasc_resnet34_384x576_f0.pt`
(fascicle, hold-out Dice 0.44). Verified in the notebook source, which calls
`find_dir("apo_resnet34_384x576_f0.pt")` and loads both files, and in its
`kernel-metadata.json`, which declares that dataset as its only `dataset_sources` entry.

**That dataset is licensed CC BY-NC-SA 4.0**, and its data card states the weights are derived from
the competition data, which carries the same licence. Two consequences, stated plainly:

1. **The non-commercial clause does not sit inside GPL-3.0.** GPL-3.0 forbids adding restrictions
   beyond its own terms, so a winning deliverable that *ships* those checkpoints could not be
   released wholly under GPL-3.0. This affects redistribution of the weights, not the right to run
   the public notebook, and it is a question for the host rather than something we can settle here.
2. **They are a third party's artifacts**, so we cannot relicense them. This is the practical
   difference from our own pipeline: there, every weight is ours and the only licence constraint is
   the one the competition data itself imposes on everyone.

**How to remove the dependency.** The weights can be retrained rather than redistributed. Training
code for exactly these two segmenters is published with the dataset (`prep.py` builds the arrays
and groups near-duplicate frames with dHash; `train_seg.py` trains with a group-aware hold-out), and
our own `train.py` trains equivalent segmenters on the same released masks, which is what U07
already does. Substituting retrained checkpoints makes every component either ours or Apache 2.0.

**What this dossier does not claim:** that an F1 rebuilt on retrained checkpoints reproduces
0.23862. That has not been measured. Measuring it needs a GPU run plus one submission, and the
substitution changes U04's segmentation, so the score would move by an unknown amount. Treat
0.23862 as the score of the entry as submitted, not of the re-runnable reconstruction.

## Re-running the combination

Each component pipeline is run first on the same ordered set of new images, each writing
`image_id,pa_deg,fl_mm,mt_mm`. Then:

```bash
python ensemble_f1.py \
  --u04 u04_predictions.csv \
  --u07 u07_predictions.csv \
  --honghanhhh honghanhhh_predictions.csv \
  --soukeaizenz soukeaizenz_predictions.csv \
  --output f1_submission.csv
```

The script requires the IDs to match exactly and rejects missing or non-finite values. There is no
per-image tuning and no leaderboard-fitted coefficient anywhere in it.

Independently checked: `ensemble_f1.py` reconstructed the submitted F1 candidate from the four
family prediction CSVs on the 309 test IDs with a maximum absolute discrepancy of **0.001** per
variable, attributable to CSV rounding alone. So the combination step is reproducible; the open
question is the component step, which is what the table above is about.

## Environment

The three Kaggle notebooks run in the Kaggle image pinned in their metadata
(`gcr.io/kaggle-images/python@sha256:17b2fc7f…`), which is what makes them re-runnable without a
dependency list of our own. Our U07 and the geometry stage run from `requirements.txt` in this
repository — except for U04's unpinned `pip install`, which the pinned image does not cover and
which would have to be pinned before any re-execution is called reproducible.

## Data

No competition images, masks, predictions or submission CSVs are distributed in this repository.
The competition data is CC BY-NC-SA and rule 2.4 forbids redistributing it. The audited submission
artifacts and their SHA-256 hashes are kept privately on the research machine.

## Summary for the final-entry decision

Of the two entries that can be marked final, this one is the ranked one. Its combination step is
reproducible today and three of its four components are re-runnable with no external artifact at
all. Its weak point is specific and now documented: one component loads third-party checkpoints
under a non-commercial licence, removable by retraining at a cost in score that has not been
measured. The companion entry from our own pipeline carries none of that: lower score, complete
code, every weight ours. The reasoning behind marking both is in the project's pending-decisions
note.
