import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from university_analytics.cli import main
from university_analytics.data import Dataset, load_dataset
from university_analytics.quality import profile_dataset, validate_dataset


def sample(rows, columns=("employee_id", "feature", "score")):
    return Dataset(columns, tuple(tuple(row) for row in rows), "unit-test-fixture", "0" * 64)


class QualityTests(unittest.TestCase):
    def test_dialect_detection_and_missing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text('id;feature;score\n1;;4\n2;2.5;5\n', encoding="utf-8")
            report = profile_dataset(load_dataset(str(path)))
        self.assertEqual(report["column_count"], 3)
        self.assertEqual(report["column_profiles"]["feature"]["missing"], 1)
        self.assertEqual(report["column_profiles"]["feature"]["min"], 2.5)

    def test_default_contract_rejects_small_fixture(self):
        report = validate_dataset(sample([["1", "2", "3"], ["2", "4", "5"]]), "score")
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["issues"]), 2)

    def test_missing_target_and_infinite_numbers_fail(self):
        dataset = sample([["1", "inf", "3"], ["2", "4", ""]])
        report = validate_dataset(dataset, "score", min_rows=2, min_columns=3)
        self.assertFalse(report["ok"])
        self.assertTrue(any("пропуски" in issue for issue in report["issues"]))
        self.assertTrue(any("бесконечность" in issue for issue in report["issues"]))

    def test_constant_numeric_target_is_rejected(self):
        report = validate_dataset(sample([["1", "2", "1"], ["2", "3", "1.0"]]), "score", min_rows=2, min_columns=3)
        self.assertFalse(report["ok"])

    def test_duplicates_counted_without_exposing_rows(self):
        dataset = sample([["secret-id", "2", "3"], ["secret-id", "2", "3"], ["2", "4", "5"]])
        report = validate_dataset(dataset, "score", min_rows=3, min_columns=3)
        self.assertTrue(report["ok"])
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertNotIn("secret-id", str(report))

    def test_cli_validation_returns_failure_for_missing_target(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text("id,score\n1,3\n2,4\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate", str(path), "--target", "missing"]), 2)
