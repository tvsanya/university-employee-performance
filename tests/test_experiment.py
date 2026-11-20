import unittest
from university_analytics.data import Dataset, DatasetError
from university_analytics.experiment import ExperimentConfig


def config(**overrides):
    values = {"task": "classification", "target": "rating", "target_description": "Метка только тестовой фикстуры", "numeric_features": ["load"], "categorical_features": ["department"], "group_column": "employee_id", "excluded_columns": ["future_score"]}
    values.update(overrides)
    return ExperimentConfig.from_dict(values)


class ExperimentTests(unittest.TestCase):
    def test_no_target_group_or_forbidden_feature_leakage(self):
        for column in ["rating", "employee_id", "future_score"]:
            with self.assertRaises(DatasetError):
                config(numeric_features=[column])

    def test_duplicate_and_missing_description_rejected(self):
        for values in [{"numeric_features": ["load", "load"]}, {"target_description": ""}, {"numeric_features": "load"}, {"seed": True}, {"unknown_option": 1}]:
            with self.assertRaises(DatasetError):
                config(**values)

    def test_columns_are_explicitly_checked(self):
        spec = config()
        data = Dataset(("rating", "load", "department", "employee_id"), (), "fixture", "0"*64)
        spec.check_columns(data)
        with self.assertRaises(DatasetError):
            spec.check_columns(Dataset(("rating",), (), "fixture", "0"*64))
        self.assertEqual(spec.features, ("load", "department"))
