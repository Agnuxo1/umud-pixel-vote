"""Dependency-free regression tests for physics_consistency_ablation.py."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from physics_consistency_ablation import (
    mean_absolute_errors,
    output_name,
    parse_alphas,
    read_numeric_csv,
    refine_length,
    run_ablation,
)


class PhysicsAblationTests(unittest.TestCase):
    """Check geometry, alignment, safety limits, and deterministic output."""

    @staticmethod
    def write_csv(path: Path, rows: list[tuple]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image_id", "pa_deg", "fl_mm", "mt_mm"))
            writer.writerows(rows)

    def test_geometry_identity_and_no_op(self) -> None:
        unchanged, exact = refine_length(30, 50, 20, 0, 0.25)
        corrected, _ = refine_length(30, 50, 20, 1, 0.25)
        self.assertAlmostEqual(exact, 40, places=8)
        self.assertAlmostEqual(unchanged, 50, places=8)
        self.assertAlmostEqual(corrected, 40, places=8)

    def test_relative_shift_cap(self) -> None:
        corrected, _ = refine_length(5, 50, 50, 1, 0.20)
        self.assertAlmostEqual(corrected, 60, places=8)

    def test_parse_alphas_validation(self) -> None:
        self.assertEqual(parse_alphas("0,0.15,1"), (0.0, 0.15, 1.0))
        for bad in ("", "0,0", "-0.1", "1.01", "not-a-number", "nan"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_alphas(bad)

    def test_file_generation_sample_order_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "predictions.csv"
            s = root / "sample.csv"
            labels = root / "labels.csv"
            self.write_csv(p, [("b", 30, 50, 20), ("a", 30, 40, 20)])
            self.write_csv(labels, [("b", 30, 40, 20), ("a", 30, 40, 20)])
            self.write_csv(s, [("a", 1, 1, 1), ("b", 1, 1, 1)])
            report = run_ablation(p, root / "out", (0.0, 1.0), 0.25, s, 2, labels, (6, 30, 5))
            self.assertEqual(report["num_images"], 2)
            self.assertEqual(len(report["candidates"]), 2)
            path = root / "out" / output_name(1.0)
            order, values = read_numeric_csv(path)
            self.assertEqual(order, ["a", "b"])
            self.assertAlmostEqual(values["b"]["fl_mm"], 40, places=5)
            self.assertAlmostEqual(
                report["candidates"][1]["heldout_validation"]["normalized_mae_supplied_tolerances"],
                0.0, places=12,
            )
            self.assertTrue((root / "out" / "physics_ablation_report.json").exists())

    def test_mismatched_sample_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, s = root / "p.csv", root / "sample.csv"
            self.write_csv(p, [("b", 30, 50, 20)])
            self.write_csv(s, [("a", 30, 50, 20)])
            with self.assertRaisesRegex(ValueError, "Sample IDs differ"):
                run_ablation(p, root / "out", sample_path=s)

    def test_duplicate_and_nonfinite_values_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            self.write_csv(path, [("a", 30, 50, 20), ("a", 31, 55, 21)])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_numeric_csv(path)
            self.write_csv(path, [("a", 30, math.nan, 20)])
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                read_numeric_csv(path)

    def test_labels_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            mean_absolute_errors({"a": {"pa_deg": 1}}, {"b": {"pa_deg": 1}})

    def test_unchanged_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "p.csv"
            self.write_csv(path, [("a", 30, 50, 20)])
            original = path.read_bytes()
            run_ablation(path, root / "out", (0.0, 0.5))
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
