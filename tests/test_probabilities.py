import math
import unittest
from university_analytics.data import DatasetError
from university_analytics.probabilities import probability_quality


class ProbabilityTests(unittest.TestCase):
    def test_hand_computed_scores_and_empty_selective_subset(self):
        report=probability_quality(['a','b'],[[.8,.2],[.6,.4]],['a','b'],bins=2,thresholds=[0,.9])
        self.assertAlmostEqual(report['log_loss'],(-math.log(.8)-math.log(.4))/2)
        self.assertAlmostEqual(report['multiclass_brier_sum'],.4)
        self.assertAlmostEqual(report['top_label_ece'],.2)
        self.assertEqual(sum(row['rows'] for row in report['reliability_bins']),2)
        self.assertIsNone(report['selective'][1]['accuracy'])
        self.assertEqual(report['selective'][1]['coverage'],0)

    def test_perfect_distribution_including_boundary_one(self):
        report=probability_quality(['a','b'],[[1,0],[0,1]],['a','b'])
        self.assertEqual(report['log_loss'],0)
        self.assertEqual(report['top_label_ece'],0)
        self.assertEqual(report['multiclass_brier_sum'],0)
        self.assertEqual(report['reliability_bins'][-1]['rows'],2)

    def test_invalid_distributions_and_unknown_classes_are_rejected(self):
        for probabilities in [[[.8,.8]], [[float('nan'),0]], [[-.1,1.1]], [[.5,.5],[.5,.5]]]:
            with self.assertRaises(DatasetError):probability_quality(['a'],probabilities,['a','b'])
        with self.assertRaises(DatasetError):probability_quality(['c'],[[.5,.5]],['a','b'])
