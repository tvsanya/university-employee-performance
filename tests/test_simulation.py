import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from university_analytics.data import load_dataset
from university_analytics.experiment import ExperimentConfig
from university_analytics.quality import validate_dataset
from university_analytics.simulation import demo_rows, generate_demo
from university_analytics.splitting import split_dataset, write_splits
from university_analytics.baselines import run_baseline_experiment


class SimulationTests(unittest.TestCase):
    def test_reproducible_rows_and_explicit_marker(self):
        first = list(demo_rows(6, 2, 42))
        self.assertEqual(first, list(demo_rows(6, 2, 42)))
        self.assertNotEqual(first, list(demo_rows(6, 2, 43)))
        self.assertEqual(len(first), 12)
        self.assertTrue(all(row[0] == "synthetic" for row in first))

    def test_origin_survives_split_training_and_saved_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            provenance = generate_demo(root/"data", employees=500, periods=2)
            data = load_dataset(str(root/"data/synthetic_employees.csv"))
            config = ExperimentConfig.from_file(root/"data/experiment.json")
            self.assertEqual(data.origin_kind, "synthetic")
            self.assertEqual(hashlib.sha256((root/"data/synthetic_employees.csv").read_bytes()).hexdigest(), provenance["sha256"])
            self.assertTrue(validate_dataset(data, config.target)["ok"])
            manifest = write_splits(data, split_dataset(data, group_column="employee_id"), root/"splits", target=config.target)
            self.assertEqual(manifest["data_origin"], "synthetic")
            report = run_baseline_experiment(root/"splits", config, root/"models")
            self.assertEqual(report["intended_use"], "development_demo_only")
            self.assertEqual(json.loads((root/"models/metrics.json").read_text())["data_origin"], "synthetic")
            self.assertNotIn("efficiency_score", config.features)
