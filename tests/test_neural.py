import unittest
import numpy as np
from university_analytics.data import Dataset, DatasetError
from university_analytics.neural import NetworkConfig, train_network
from university_analytics.preprocessing import feature_matrix
from tests.test_baselines import fixture


class NeuralTests(unittest.TestCase):
    def test_external_validation_early_stop_and_best_checkpoint(self):
        _, config, _, parts = fixture(size=80)
        parameters = NetworkConfig(hidden_layers=(8, 4), max_epochs=20, patience=2, min_delta=1)
        model, report = train_network(parts["train"], parts["validation"], config, parameters)
        self.assertEqual(report["stop_reason"], "patience")
        best = max(item["validation"]["macro_f1"] for item in report["history"])
        self.assertEqual(report["best_validation_macro_f1"], best)
        self.assertEqual(report["architecture"]["hidden_layers"], [8, 4])
        self.assertEqual(report["architecture"]["output_units"], 1)
        self.assertEqual(len(model.predict(feature_matrix(parts["test"], config))), len(parts["test"].rows))

    def test_fixed_seed_reproduces_history_and_weights(self):
        _, config, _, parts = fixture(size=80)
        parameters = NetworkConfig(hidden_layers=(8, 4), max_epochs=5)
        first, first_report = train_network(parts["train"], parts["validation"], config, parameters)
        second, second_report = train_network(parts["train"], parts["validation"], config, parameters)
        self.assertEqual(first_report, second_report)
        for left, right in zip(first.named_steps["model"].coefs_, second.named_steps["model"].coefs_):
            np.testing.assert_array_equal(left, right)

    def test_bad_architectures_and_nonfinite_parameters_rejected(self):
        for values in [{"hidden_layers": [8]}, {"hidden_layers": [8, 0]}, {"learning_rate": float("nan")},
                       {"max_epochs": True}, {"seed": -1}, {"unknown": 1}, {"balanced": "yes"}]:
            with self.assertRaises(DatasetError):
                NetworkConfig.from_dict(values)

    def test_unknown_validation_class_and_regression_rejected(self):
        _, config, _, parts = fixture()
        validation = parts["validation"]
        unknown = Dataset(validation.columns, tuple((row[0], "new-class", *row[2:]) for row in validation.rows), "fixture", "0"*64)
        with self.assertRaises(DatasetError):
            train_network(parts["train"], unknown, config)
        _, regression, _, regression_parts = fixture("regression")
        with self.assertRaises(DatasetError):
            train_network(regression_parts["train"], regression_parts["validation"], regression)
