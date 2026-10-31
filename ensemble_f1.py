"""Recreate the UMUD F1 ensemble from four independently generated model outputs.

F1 uses the arithmetic mean of three independently trained model families:
1. A two-model own family, formed by averaging U04 and U07.
2. The cited Honghanhhh public Kaggle notebook predictions.
3. The cited Soukeaizenz public Kaggle notebook predictions.

The software accepts unseen images via their model inference CSVs. No labels,
test targets, leaderboard scores, or per-image fitted constants are required.
Original raw data and third-party predictions are intentionally not redistributed.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

TARGETS = ("pa_deg", "fl_mm", "mt_mm")
COLUMNS = ("image_id",) + TARGETS


def read_predictions(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Load a Kaggle prediction CSV, preserving its row order and validating all values."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        delimiter = ";" if header.count(";") > header.count(",") else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if set(COLUMNS) - set(reader.fieldnames or ()):
            raise ValueError(f"Missing prediction fields in {path}")
        order: list[str] = []
        results: dict[str, dict[str, float]] = {}
        for row in reader:
            identity = row["image_id"].strip()
            if identity in results or not identity:
                raise ValueError(f"Duplicate/empty image identifier in {path}: {identity!r}")
            numbers = {field: float(row[field]) for field in TARGETS}
            if any(not math.isfinite(value) for value in numbers.values()):
                raise ValueError(f"Non-finite prediction for {identity} in {path}")
            order.append(identity)
            results[identity] = numbers
    if not order:
        raise ValueError(f"Empty prediction file: {path}")
    return order, results


def build_f1(u04: Path, u07: Path, honghanhhh: Path, soukeaizenz: Path, output: Path) -> None:
    """Produce F1 with fixed, uniform family weights and no leaderboard optimization."""
    order, a = read_predictions(u04)
    _, b = read_predictions(u07)
    _, c = read_predictions(honghanhhh)
    _, d = read_predictions(soukeaizenz)
    expected = set(order)
    for label, source in (("U07", b), ("Honghanhhh", c), ("Soukeaizenz", d)):
        if set(source) != expected:
            raise ValueError(f"Image identifiers do not match for {label}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for identity in order:
            row = {"image_id": identity}
            for field in TARGETS:
                own_family = 0.5 * (a[identity][field] + b[identity][field])
                ensemble = (own_family + c[identity][field] + d[identity][field]) / 3.0
                row[field] = f"{ensemble:.3f}"
            writer.writerow(row)
    print(f"F1 ensemble generated: {output} ({len(order)} images)")


def main() -> None:
    """Resolve supplied CSV paths and generate a fully deterministic F1 submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u04", type=Path, required=True)
    parser.add_argument("--u07", type=Path, required=True)
    parser.add_argument("--honghanhhh", type=Path, required=True)
    parser.add_argument("--soukeaizenz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_f1(args.u04, args.u07, args.honghanhhh, args.soukeaizenz, args.output)


if __name__ == "__main__":
    main()
