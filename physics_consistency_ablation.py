"""Create conservative, physically consistent UMUD ensemble ablations.

This script DOES NOT optimize against Kaggle's public leaderboard. It produces
predeclared candidates that can be evaluated on genuinely independent labelled
validation data, if such data are available. The original predictions are never
modified in place. No competition data or model weights are distributed here.

The approximation FL = MT / sin(PA) assumes locally parallel aponeuroses and
approximately straight fascicles. Actual clinical annotations may deviate from
this idealization. Therefore the default adjustment is capped at 25% of the
original fascicle-length prediction, and a candidate must NOT be submitted
without a defensible out-of-domain validation result.

Examples
--------
python physics_consistency_ablation.py --predictions base.csv --output-dir ablations
python physics_consistency_ablation.py --predictions base.csv --output-dir ablations \
    --sample sample_submission.csv --expected-rows 309 --alphas 0,0.15,0.3,0.5
python physics_consistency_ablation.py --predictions holdout_predictions.csv \
    --output-dir holdout_ablation --labels disjoint_holdout_labels.csv

To compute a normalized score as well, specify three tolerances only after
verifying them against the current official evaluation implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

COLUMNS = ("image_id", "pa_deg", "fl_mm", "mt_mm")
TARGETS = COLUMNS[1:]
DEFAULT_ALPHAS = (0.0, 0.15, 0.30, 0.50)


def read_numeric_csv(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Read and validate a three-target CSV with unique IDs and finite numbers."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != set(COLUMNS):
            raise ValueError(f"{path}: expected exactly these columns: {', '.join(COLUMNS)}")
        order: list[str] = []
        values: dict[str, dict[str, float]] = {}
        for line_number, record in enumerate(reader, start=2):
            identifier = (record.get("image_id") or "").strip()
            if not identifier or identifier in values:
                raise ValueError(f"{path}:{line_number}: missing or duplicate image_id")
            try:
                targets = {key: float(record[key]) for key in TARGETS}
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: nonnumeric target") from error
            if not all(math.isfinite(value) for value in targets.values()):
                raise ValueError(f"{path}:{line_number}: nonfinite target")
            if targets["pa_deg"] <= 0 or targets["fl_mm"] <= 0 or targets["mt_mm"] <= 0:
                raise ValueError(f"{path}:{line_number}: targets must be strictly positive")
            order.append(identifier)
            values[identifier] = targets
    if not order:
        raise ValueError(f"{path}: the CSV contains no image records")
    return order, values


def read_sample_ids(path: Path) -> list[str]:
    """Read image IDs from the official sample without trusting prediction values."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise ValueError(f"{path}: sample requires an image_id column")
        identifiers = [(row.get("image_id") or "").strip() for row in reader]
    if not identifiers or any(not identifier for identifier in identifiers):
        raise ValueError(f"{path}: empty sample or empty image IDs")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: duplicate sample identifiers")
    return identifiers


def parse_alphas(text: str) -> tuple[float, ...]:
    """Accept a unique list of blend fractions in the closed interval [0, 1]."""
    try:
        alphas = tuple(float(piece.strip()) for piece in text.split(","))
    except ValueError as error:
        raise ValueError("Alphas must be comma-separated floating-point numbers") from error
    if not alphas or any(not math.isfinite(a) or not 0.0 <= a <= 1.0 for a in alphas):
        raise ValueError("Every alpha must lie between 0 and 1")
    if len(set(alphas)) != len(alphas):
        raise ValueError("Duplicate alpha values are not allowed")
    return alphas


def refine_length(
    pa_deg: float,
    fl_mm: float,
    mt_mm: float,
    alpha: float,
    max_relative_shift: float,
) -> tuple[float, float]:
    """Blend FL toward its geometric counterpart with a bounded correction.

    Returns (new_fascicle_length, unbounded_geometric_fascicle_length).
    Neither PA nor MT is changed. A tiny positive-angle floor prevents overflow;
    predictions outside the official physiology range are flagged elsewhere,
    not silently changed by this function.
    """
    if not (0.0 <= alpha <= 1.0 and 0.0 <= max_relative_shift <= 1.0):
        raise ValueError("alpha and max_relative_shift must be in [0, 1]")
    sine = math.sin(math.radians(max(1e-6, min(89.999999, pa_deg))))
    geometric_length = mt_mm / sine
    capped_geometric_length = min(200.0, max(30.0, geometric_length))
    limit = fl_mm * max_relative_shift
    correction = max(-limit, min(limit, capped_geometric_length - fl_mm))
    new_length = fl_mm + alpha * correction
    return new_length, geometric_length


def mean_absolute_errors(
    predictions: dict[str, dict[str, float]],
    labels: dict[str, dict[str, float]],
    tolerances: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """Evaluate ONLY supplied, genuinely held-out labelled data, by aligned ID."""
    if set(predictions) != set(labels):
        raise ValueError("Prediction and label identifiers must match exactly")
    mae = {
        key: mean(abs(predictions[identifier][key] - labels[identifier][key])
                  for identifier in predictions)
        for key in TARGETS
    }
    if tolerances is not None:
        if len(tolerances) != 3 or any(not math.isfinite(x) or x <= 0 for x in tolerances):
            raise ValueError("Tolerances must be exactly three strictly positive finite numbers")
        mae["normalized_mae_supplied_tolerances"] = mean(
            mae[key] / tolerance for key, tolerance in zip(TARGETS, tolerances)
        )
    return mae


def output_name(alpha: float) -> str:
    """Name an ablation without introducing locale-dependent decimal separators."""
    return "submission_physics_a" + f"{alpha:.4f}".replace(".", "p") + ".csv"


def run_ablation(
    predictions_path: Path,
    output_dir: Path,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    max_relative_shift: float = 0.25,
    sample_path: Path | None = None,
    expected_rows: int | None = None,
    labels_path: Path | None = None,
    tolerances: tuple[float, float, float] | None = None,
) -> dict:
    """Validate IDs, generate candidates, and return a machine-readable audit."""
    alphas = parse_alphas(",".join(str(a) for a in alphas))
    if not math.isfinite(max_relative_shift) or not 0.0 <= max_relative_shift <= 1.0:
        raise ValueError("max_relative_shift must lie in [0, 1]")
    order, original = read_numeric_csv(predictions_path)
    if expected_rows is not None and len(order) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; found {len(order)}")
    if sample_path is not None:
        sample_order = read_sample_ids(sample_path)
        if set(sample_order) != set(order):
            missing = set(sample_order) - set(order)
            surplus = set(order) - set(sample_order)
            raise ValueError(f"Sample IDs differ: missing={len(missing)}, surplus={len(surplus)}")
        order = sample_order
    labels = None
    if labels_path is not None:
        _, labels = read_numeric_csv(labels_path)
        if set(labels) != set(order):
            raise ValueError("Validation labels do not align with prediction IDs")
    if tolerances is not None and labels is None:
        raise ValueError("--tolerances requires --labels")

    outside_ranges = {
        "pa_deg": sum(not 5 <= original[k]["pa_deg"] <= 45 for k in order),
        "fl_mm": sum(not 30 <= original[k]["fl_mm"] <= 200 for k in order),
        "mt_mm": sum(not 10 <= original[k]["mt_mm"] <= 50 for k in order),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "prediction_source": str(predictions_path),
        "prediction_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "num_images": len(order),
        "outside_official_test_ranges": outside_ranges,
        "method": "bounded blend of predicted FL toward MT/sin(PA)",
        "max_relative_shift": max_relative_shift,
        "validation_claim": "no hidden labels or leaderboard data used",
        "candidates": [],
    }
    for alpha in alphas:
        candidate = {}
        changes = []
        geometry_gaps = []
        for image_id in order:
            values = original[image_id]
            fl_new, fl_geometry = refine_length(
                values["pa_deg"], values["fl_mm"], values["mt_mm"],
                alpha, max_relative_shift,
            )
            candidate[image_id] = {
                "pa_deg": values["pa_deg"],
                "fl_mm": fl_new,
                "mt_mm": values["mt_mm"],
            }
            changes.append(abs(fl_new - values["fl_mm"]))
            geometry_gaps.append(abs(fl_geometry - values["fl_mm"]))
        csv_path = output_dir / output_name(alpha)
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for image_id in order:
                writer.writerow({"image_id": image_id, **{
                    key: f"{candidate[image_id][key]:.6f}" for key in TARGETS
                }})
        entry = {
            "alpha": alpha,
            "file": str(csv_path),
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "mean_absolute_fl_change_mm": mean(changes),
            "mean_absolute_unconstrained_geometry_gap_mm": mean(geometry_gaps),
        }
        if labels is not None:
            entry["heldout_validation"] = mean_absolute_errors(candidate, labels, tolerances)
        report["candidates"].append(entry)
    report_path = output_dir / "physics_ablation_report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    """Parse CLI arguments and produce independent, fully traceable CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alphas", default="0,0.15,0.30,0.50")
    parser.add_argument("--max-relative-shift", default=0.25, type=float)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--tolerances", help="PA,FL,MT tolerances from verified evaluator")
    args = parser.parse_args()
    try:
        tolerances = None
        if args.tolerances:
            parts = tuple(float(item.strip()) for item in args.tolerances.split(","))
            if len(parts) != 3:
                raise ValueError("Exactly three tolerances are required")
            tolerances = parts
        report = run_ablation(
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            alphas=parse_alphas(args.alphas),
            max_relative_shift=args.max_relative_shift,
            sample_path=args.sample,
            expected_rows=args.expected_rows,
            labels_path=args.labels,
            tolerances=tolerances,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
