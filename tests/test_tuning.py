import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from university_analytics.cli import main
from university_analytics.data import Dataset, DatasetError
from university_analytics.experiment import ExperimentConfig
from university_analytics.neural import NetworkConfig
from university_analytics.simulation import generate_demo
from university_analytics.splitting import split_dataset, write_splits
from university_analytics.data import load_dataset
from university_analytics.tuning import fit_neural_candidates, load_network_grid
from tests.test_baselines import fixture


class TuningTests(unittest.TestCase):
    def test_test_features_and_labels_do_not_change_selection_or_weights(self):
        _, config, _, datasets = fixture(size=80)
        candidates = [NetworkConfig(hidden_layers=(8, 4), max_epochs=4),
                      NetworkConfig(hidden_layers=(6, 3), max_epochs=4)]
        first, report, history = fit_neural_candidates(datasets, config, candidates)
        test = datasets["test"]
        changed = dict(datasets)
        changed["test"] = Dataset(test.columns, tuple((row[0], "1" if row[1] == "0" else "0", *["999999"]*8) for row in test.rows), "fixture", "0"*64)
        second, other_report, other_history = fit_neural_candidates(changed, config, candidates)
        self.assertEqual(report["selected_candidate"], other_report["selected_candidate"])
        self.assertEqual(history, other_history)
        for left, right in zip(first.named_steps["model"].coefs_, second.named_steps["model"].coefs_):
            np.testing.assert_array_equal(left, right)
        self.assertNotIn("test", history["trials"][0]["history"][0])

    def test_invalid_and_duplicate_grids_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"grid.json"
            for values in [[], [{}]*9, [{}, {}], [{"activation": "unknown"}], ["bad"], {}]:
                path.write_text(json.dumps(values))
                with self.assertRaises(DatasetError):
                    load_network_grid(path)

    def test_cli_saves_simulation_origin_history_and_reloadable_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate_demo(root/"data", employees=500, periods=2)
            data = load_dataset(str(root/"data/synthetic_employees.csv"))
            config = ExperimentConfig.from_file(root/"data/experiment.json")
            split = split_dataset(data, group_column="employee_id")
            write_splits(data, split, root/"split", target=config.target)
            (root/"grid.json").write_text(json.dumps([{"hidden_layers": [8, 4], "max_epochs": 3}]))
            args = ["train-neural", "--splits", str(root/"split"), "--config", str(root/"data/experiment.json"),
                    "--grid", str(root/"grid.json"), "--output", str(root/"model")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads((root/"model/metrics.json").read_text())
            self.assertEqual((report["data_origin"], report["intended_use"]), ("synthetic", "development_demo_only"))
            self.assertIn("scikit-learn", report["versions"])
            self.assertEqual(len(json.loads((root/"model/training.json").read_text())["trials"]), 1)
            model = joblib.load(root/"model/model3.joblib")
            self.assertEqual(len(model.named_steps["model"].coefs_), 3)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
