import unittest
import numpy as np
from university_analytics.data import Dataset, DatasetError
from university_analytics.experiment import ExperimentConfig
from university_analytics.preprocessing import fit_preprocessor, transform_features, target_vector


def dataset(rows, columns=("x", "department", "target")):
    return Dataset(columns, tuple(tuple(row) for row in rows), "unit-test-fixture", "0"*64)


def spec(task="classification"):
    return ExperimentConfig(task, "target", "Цель тестовой фикстуры", ("x",), ("department",))


class PreprocessingTests(unittest.TestCase):
    def test_heldout_does_not_change_train_statistics_or_vocabulary(self):
        config = spec()
        train = dataset([["1", "A", "low"], ["3", "A", "high"], ["", "A", "low"]])
        preprocess = fit_preprocessor(train, config)
        numeric = preprocess.named_transformers_["numeric"]
        self.assertEqual(numeric.named_steps["impute"].statistics_[0], 2)
        self.assertEqual(numeric.named_steps["scale"].mean_[0], 2)
        heldout = dataset([["100", "B", "high"]])
        result = transform_features(heldout, config, preprocess)
        self.assertEqual(numeric.named_steps["scale"].mean_[0], 2)
        self.assertEqual(preprocess.named_transformers_["categorical"].named_steps["encode"].categories_[0].tolist(), ["A"])
        self.assertEqual(result[0, -1], 0)

    def test_prediction_can_omit_target_and_reorder_columns(self):
        config = spec()
        preprocess = fit_preprocessor(dataset([["1", "A", "low"], ["3", "B", "high"]]), config)
        result = transform_features(dataset([["A", "2"]], ("department", "x")), config, preprocess)
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(np.isfinite(result).all())

    def test_fully_missing_features_are_imputed_without_using_heldout(self):
        config = spec()
        preprocess = fit_preprocessor(dataset([["", "", "low"], ["", "", "high"]]), config)
        output = transform_features(dataset([["5", "new", "low"]]), config, preprocess)
        self.assertTrue(np.isfinite(output).all())

    def test_invalid_numeric_features_and_targets_rejected(self):
        for value in ["inf", "text"]:
            with self.assertRaises(DatasetError):
                fit_preprocessor(dataset([[value, "A", "low"]]), spec())
        for target in ["", "NaN", "inf", "text"]:
            with self.assertRaises(DatasetError):
                target_vector(dataset([["1", "A", target]]), spec("regression"))
