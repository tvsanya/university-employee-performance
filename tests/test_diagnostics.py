import json
import tempfile
import unittest
from pathlib import Path
from university_analytics.data import DatasetError
from university_analytics.diagnostics import classification_diagnostics, run_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_matrix_support_and_unknown_true_class_are_not_dropped(self):
        report = classification_diagnostics(["a", "b", "c", "a"], ["a", "a", "b", "a"], ["a", "b"])
        self.assertEqual(sum(sum(row) for row in report["confusion_matrix"]), 4)
        self.assertEqual(report["error_count"], 2)
        self.assertEqual(sum(item["support"] for item in report["per_class"]), 4)
        self.assertIn("c", report["labels"])
        self.assertEqual(report["error_rows"][0]["source_row"], 3)
        self.assertEqual(json.loads(json.dumps(report))["error_count"], 2)

    def test_empty_and_mismatched_vectors_rejected(self):
        for truth, predicted in [([], []), (["a"], [])]:
            with self.assertRaises(DatasetError):
                classification_diagnostics(truth, predicted, ["a", "b"])

    def test_artifact_path_cannot_escape_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"metrics.json").write_text(json.dumps({"portable_artifact": {"file": "../outside.json"}}))
            with self.assertRaises(DatasetError):
                run_diagnostics(root/"splits", root, root/"diagnostics")
