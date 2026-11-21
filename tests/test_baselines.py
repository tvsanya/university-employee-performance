import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from university_analytics.baselines import fit_baselines, load_split_bundle, scores
from university_analytics.cli import main
from university_analytics.data import Dataset, DatasetError
from university_analytics.experiment import ExperimentConfig
from university_analytics.preprocessing import feature_matrix
from university_analytics.splitting import split_dataset, write_splits


def fixture(task="classification", size=60):
    # Программные фикстуры не являются данными дипломного исследования.
    columns = ("employee_id", "target", *[f"x{j}" for j in range(8)])
    rows = tuple((str(n), str(n % 2 if task == "classification" else n*2), *[str(n+j) for j in range(8)]) for n in range(size))
    dataset = Dataset(columns, rows, "unit-test-fixture", "0"*64)
    config = ExperimentConfig(task, "target", "Исключительно тест конвейера", tuple(f"x{j}" for j in range(8)), group_column="employee_id")
    split = split_dataset(dataset, group_column="employee_id")
    return dataset, config, split, {name: Dataset(columns, rows, "unit-test-fixture", "0"*64) for name, rows in split.partitions.items()}


class BaselineTests(unittest.TestCase):
    def test_missing_class_keeps_its_zero_contribution_in_macro_f1(self):
        result = scores("classification", ["a", "a"], ["a", "a"], labels=["a", "b"])
        self.assertEqual((result["accuracy"], result["macro_f1"]), (1, .5))

    def test_classification_metrics_and_validation_selection(self):
        _, config, _, datasets = fixture()
        models, report = fit_baselines(datasets, config)
        self.assertEqual(set(models), {"dummy", "linear"})
        self.assertEqual(report["selection_partition"], "validation")
        for model in report["metrics"].values():
            for part in model.values():
                self.assertTrue(0 <= part["accuracy"] <= 1)
                self.assertTrue(0 <= part["macro_f1"] <= 1)

    def test_regression_models_do_not_learn_from_changed_heldout_targets(self):
        _, config, _, datasets = fixture("regression")
        first, _ = fit_baselines(datasets, config)
        changed = dict(datasets)
        for name in ["validation", "test"]:
            data = changed[name]
            changed[name] = Dataset(data.columns, tuple((row[0], "999999", *row[2:]) for row in data.rows), data.source, data.sha256)
        second, _ = fit_baselines(changed, config)
        matrix = feature_matrix(datasets["test"], config)
        for name in first:
            np.testing.assert_allclose(first[name].predict(matrix), second[name].predict(matrix))

    def test_tampered_split_is_rejected_before_training(self):
        dataset, config, result, _ = fixture(size=1000)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"split"
            write_splits(dataset, result, path, target="target")
            load_split_bundle(path, config)
            target = path/"test.csv"
            target.write_text(target.read_text()+"broken\n")
            with self.assertRaises(DatasetError):
                load_split_bundle(path, config)

    def test_cli_saves_reloadable_models_with_exact_versions(self):
        dataset, config, result, parts = fixture(size=1000)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_splits(dataset, result, root/"split", target="target")
            (root/"config.json").write_text(json.dumps(config.as_dict()))
            args = ["train-baseline", "--splits", str(root/"split"), "--config", str(root/"config.json"), "--output", str(root/"experiment")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads((root/"experiment/metrics.json").read_text())
            self.assertIn("scikit-learn", report["versions"])
            saved = report["artifacts"][report["selected_model"]]["file"]
            model = joblib.load(root/"experiment"/saved)
            predicted = model.predict(feature_matrix(parts["test"], config))
            self.assertEqual(len(predicted), len(parts["test"].rows))
