import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from university_analytics.data import DatasetError
from university_analytics.prediction import run_prediction


class ReviewPolicyTests(unittest.TestCase):
    def test_boundary_entropy_flags_and_unchanged_classes(self):
        model = SimpleNamespace(classes=["a", "b", "c"], sha256="0"*64,
            data_origin="synthetic", intended_use="development_demo_only",
            config=SimpleNamespace(group_column=None),
            predict_proba=lambda dataset: np.array([[.7,.2,.1],[.4,.35,.25]]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source=root/"input.csv"; source.write_text("x\n1\n2\n")
            with patch("university_analytics.prediction.PortableNetwork.from_file", return_value=model):
                report = run_prediction("unused", str(source), root/"result", review_threshold=.7)
            self.assertEqual(report["review_policy"]["review_rows"], 1)
            self.assertEqual(report["review_policy"]["coverage_above_threshold"], .5)
            with (root/"result/predictions.csv").open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual([row["review_recommended"] for row in rows], ["no","yes"])
            self.assertEqual([row["predicted_class"] for row in rows], ["a","a"])
            self.assertGreater(float(rows[1]["normalized_entropy"]), float(rows[0]["normalized_entropy"]))

    def test_nonfinite_and_out_of_bounds_threshold_rejected_before_output(self):
        for value in [float("nan"), float("inf"), -1, 1.1, True, ".7"]:
            with self.assertRaises(DatasetError): run_prediction("unused", "unused", "unused", review_threshold=value)
