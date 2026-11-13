# UMUD final-window execution protocol

**Competition:** [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)  
**Objective:** maximize generalization on the hidden private test partition and produce a prize-eligible, end-to-end reproducible submission.  
**Prepared:** 13 November 2026.  
**Expected close from existing project records:** 14 November 2026, 23:00 UTC (15 November, 00:00 in Madrid). **Reconfirm against live Kaggle before final submissions.**

## Evidence and starting point

The latest local execution evidence recovered from the earlier 13 November audit reported **50 of 195 evaluations** completed, with an RTX 3090 training supervisor running. That state is not freshly verified here: Desktop Commander returned a monthly usage-limit error. Do not reconnect or repeatedly retry.

The strongest previously reported public result was **0.22941**, around **rank 60/658** on 12 November, pending live Kaggle verification. The published F1 benchmark (submission 57852101) scored **0.23862** on 31 October. Neither is a verified private-score result or evidence of winning. Retain the artifact, configuration, code commit and SHA-256 for the most recent verified champion.

The organizer reports **670 exact duplicate fascicle image/mask pairs** and warns that naïve validation can be optimistic. The public leaderboard uses approximately **33%** of test instances; final rank is based on the other **67%**.

## Gate 1 — Preserve working GPU execution

Do **not** restart a possibly live training supervisor just to gather status. On an authorized local session, first inspect existing process trees, GPU reservations, current evaluation number, latest checkpoint, log timestamps and disk space. Avoid flashing Python console windows or spawning duplicate supervisors. Resume the existing queue only if telemetry shows a genuine stall and recovery is safe.

The user has reserved other GPUs for unrelated projects: do not steal a reserved GPU or displace an existing run. Run UMUD only on its designated available RTX 3090.

## Gate 2 — Assess out-of-domain evidence

- Keep train/selection/report sets disjoint by participant or acquisition session where metadata allow. Frame size alone is not proof of scanner identity.
- Deduplicate fascicle image/mask pairs before generating folds, including cross-fold hash and near-duplicate checks.
- Compare the current champion, independent U04/U07 family predictions, U-Net and NeuroPixel experiments on the **same** held-out IDs using the **official** metric. Track per-target MAE, not only pooled loss.
- Separate oracle-mask orientation studies from fully automatic inference. The former cannot establish test-time transfer.
- Prioritize whichever parameter dominates error, especially scanner-dependent physical-scale failures. Avoid recycling a poor standalone hybrid merely because its oracle-mask angle metric was strong.
- Distinguish out-of-domain improvement from run-to-run noise using fixed splits, multiple seeds and paired per-image differences.

## Gate 3 — Test physical consistency without leaderboard fitting

This repository includes physics_consistency_ablation.py, which creates independent, **bounded** fascicle-length candidate corrections at alpha 0, 0.15, 0.30 and 0.50. It optionally measures error on **genuinely held-out labelled** data.

The approximation **FL ≈ MT / sin(PA)** assumes locally parallel aponeuroses and straight fascicles. It is not exact for all clinical annotations. Do not submit these candidates without held-out support. The alpha=0 control preserves the numerical predictions.

Example after identifying the correct champion and official sample CSV on the research PC:

    python physics_consistency_ablation.py --predictions champion.csv --output-dir physics_candidates --sample sample_submission.csv --expected-rows 309

For labelled independent training validation, the optional --labels argument adds per-target MAE. Pass --tolerances only after verifying official constants from the current evaluator. The tool never modifies model weights or the original submission file.

## Gate 4 — Spend submissions only on evidence

Check Kaggle's current daily submission quota and any already-queued submissions. Every submission must satisfy:

1. Exact CSV columns image_id,pa_deg,fl_mm,mt_mm; exactly 309 unique aligned test IDs for the present set; finite numeric predictions.
2. Reproducible inference on fresh ultrasound images, with fixed weights, preprocessing, code revision and documented external dependencies.
3. Measured out-of-domain support or a rigorously justified new hypothesis. Do not optimize blend coefficients on the public leaderboard.
4. Recorded checkpoint, commit, CSV SHA-256, rationale, Kaggle submission ID, public score and timestamp.
5. The previous champion remains preserved even if a new test score is worse.

## Gate 5 — Final ranking and award eligibility

The organizers require prize-eligible methods to release an **OSI-compatible open-source repository**, FAIR metadata, instructions, environment specification and runnable inference pipeline. This repository chooses GPL-3.0, but the competition does **not** mandate GPL-3.0 specifically.

The documented F1 blend includes U04, U07 and two external inference notebooks. Its averaging script alone is not complete reproducibility. The U04 third-party weights use a non-commercial data licence, and the re-execution dossier records package-version and notebook-version gaps. Resolve these before calling that submission award-ready; do not redistribute restricted data or third-party weights.

The decisive organizer test is whether the final method runs on **new ultrasound images without a leaderboard**.

Select the best allowable final Kaggle submissions before the confirmed closing time and verify the correct entries are selected as final. Never confuse a public rank with private final standing.

## Scientific integrity

No hidden labels, manual labeling of withheld test cases, training on copied evaluation targets, public-score fitted parameters, fabricated experiments or unsupported efficiency claims. Every reported improvement must be traced to a real measurement with a reproducible run.

**Document status:** operational decision protocol and a newly testable hypothesis; it is **not** proof of a new Kaggle submission, higher score or a winning rank.
