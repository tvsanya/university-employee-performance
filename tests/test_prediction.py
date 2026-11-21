import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from university_analytics.cli import main
from university_analytics.data import Dataset, DatasetError
from university_analytics.experiment import ExperimentConfig
from university_analytics.neural import NetworkConfig, train_network
from university_analytics.portable import export_network, PortableNetwork
from university_analytics.preprocessing import feature_matrix
from university_analytics.simulation import COLUMNS, demo_rows
from university_analytics.splitting import split_dataset
from tests.test_baselines import fixture


class PredictionTests(unittest.TestCase):
    def test_binary_json_probabilities_match_pipeline(self):
        _, config, _, parts = fixture(size=80)
        model, _ = train_network(parts["train"], parts["validation"], config,
            NetworkConfig(hidden_layers=(8, 4), max_epochs=4))
        portable = PortableNetwork(export_network(model, config))
        expected = model.predict_proba(feature_matrix(parts["test"], config))
        np.testing.assert_allclose(portable.predict_proba(parts["test"]), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(portable.predict(parts["test"]), model.predict(feature_matrix(parts["test"], config)))

    def simulation_model(self):
        data = Dataset(COLUMNS, tuple(demo_rows(employees=100, periods=2)), "synthetic", "0"*64, "synthetic")
        config = ExperimentConfig("classification", "efficiency_level", "Тест JSON-конвейера",
            ("student_feedback_avg", "publication_count"), ("department",), "employee_id", ("efficiency_score",))
        split = split_dataset(data, group_column="employee_id")
        parts = {name: Dataset(COLUMNS, rows, "synthetic", "0"*64, "synthetic") for name, rows in split.partitions.items()}
        model, _ = train_network(parts["train"], parts["validation"], config,
            NetworkConfig(hidden_layers=(8, 4), max_epochs=4))
        return model, config, parts

    def test_multiclass_missing_unknown_categories_and_no_target(self):
        model, config, parts = self.simulation_model()
        columns = ("employee_id", "student_feedback_avg", "publication_count", "department")
        data = Dataset(columns, (("SIM-NEW", "", "", "Неизвестная"), ("SIM-NEW2", "4.2", "2", "")), "fixture", "0"*64)
        portable = PortableNetwork(export_network(model, config, "synthetic"))
        np.testing.assert_allclose(portable.predict_proba(data), model.predict_proba(feature_matrix(data, config)), atol=1e-12)
        self.assertEqual(portable.intended_use, "development_demo_only")

    def test_invalid_schema_weights_and_metadata_are_rejected(self):
        model, config, _ = self.simulation_model()
        original = export_network(model, config, "synthetic")
        changes = [("format", "pickle"), ("data_origin", []), ("intended_use", "research"),
                   ("numeric", {"impute": [1, 2], "mean": [0, 0], "scale": [0, 1]}),
                   ("network", {**original["network"], "weights": []}),
                   ("network", {**original["network"], "activation": []})]
        for key, value in changes:
            payload = copy.deepcopy(original)
            payload[key] = value
            with self.assertRaises(DatasetError):
                PortableNetwork(payload)
        payload = copy.deepcopy(original)
        payload["network"]["weights"][0][0][0] = float("nan")
        with self.assertRaises(DatasetError):
            PortableNetwork(payload)
        with self.assertRaises(DatasetError):
            ExperimentConfig.from_dict({**config.as_dict(), "task": []})
        with self.assertRaises(DatasetError):
            NetworkConfig.from_dict({"activation": []})

    def test_cli_predicts_without_retraining_and_preserves_origin(self):
        model, config, _ = self.simulation_model()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"model.json").write_text(json.dumps(export_network(model, config, "synthetic")))
            (root/"input.csv").write_text("employee_id,student_feedback_avg,publication_count,department\nSIM-NEW,4.2,2,Новая\n")
            args = ["predict", str(root/"input.csv"), "--model", str(root/"model.json"), "--output", str(root/"predictions")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads((root/"predictions/report.json").read_text())
            self.assertEqual((report["rows"], report["data_origin"], report["intended_use"]), (1, "synthetic", "development_demo_only"))
            with (root/"predictions/predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["employee_id"], "SIM-NEW")
            self.assertAlmostEqual(sum(float(value) for key, value in rows[0].items() if key.startswith("probability_")), 1, places=10)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
