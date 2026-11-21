import json
import tempfile
import unittest
from pathlib import Path
from university_analytics.comparison import compare_experiments
from university_analytics.data import DatasetError
from tests.test_baselines import fixture


class ComparisonTests(unittest.TestCase):
    def make_reports(self, root):
        _, config, _, _ = fixture()
        directories = []
        for name, validation, test in [("linear", .8, .4), ("neural", .7, .99)]:
            path = root/name;path.mkdir();directories.append(path)
            (path/"config.json").write_text(json.dumps(config.as_dict()))
            report = {"source_sha256": "a"*64, "split_manifest_sha256": "b"*64, "data_origin": "synthetic", "rows": {"train": 10},
                "metrics": {name: {part: {"macro_f1": value} for part, value in [("train", .9), ("validation", validation), ("test", test)]}}}
            (path/"metrics.json").write_text(json.dumps(report))
        return directories

    def test_high_test_score_cannot_win_over_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            report = compare_experiments(self.make_reports(Path(folder)))
            self.assertEqual(report["selected"], "1:linear")
            self.assertEqual(report["intended_use"], "development_demo_only")

    def test_mismatched_data_and_duplicate_experiments_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.make_reports(Path(folder))
            with self.assertRaises(DatasetError):
                compare_experiments([paths[0], paths[0]])
            report = json.loads((paths[1]/"metrics.json").read_text());report["source_sha256"] = "c"*64
            (paths[1]/"metrics.json").write_text(json.dumps(report))
            with self.assertRaises(DatasetError):
                compare_experiments(paths)
