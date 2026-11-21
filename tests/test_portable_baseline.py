import unittest
import numpy as np
from university_analytics.baselines import fit_baselines
from university_analytics.portable import export_baseline, PortableNetwork
from university_analytics.preprocessing import feature_matrix
from tests.test_baselines import fixture


class PortableBaselineTests(unittest.TestCase):
    def test_linear_and_dummy_predictions_match_saved_pipeline(self):
        _, config, _, parts = fixture(size=100)
        models, report = fit_baselines(parts, config)
        for model in models.values():
            portable = PortableNetwork(export_baseline(model, config, "synthetic"))
            matrix = feature_matrix(parts["test"], config)
            np.testing.assert_allclose(portable.predict_proba(parts["test"]), model.predict_proba(matrix), atol=1e-12)
            np.testing.assert_array_equal(portable.predict(parts["test"]), model.predict(matrix))
            self.assertEqual(portable.intended_use, "development_demo_only")
