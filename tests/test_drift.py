import unittest
from university_analytics.data import Dataset
from university_analytics.experiment import ExperimentConfig
from university_analytics.drift import distribution_shift, numeric_distance
import json


class DistributionTests(unittest.TestCase):
    def test_identical_distributions_have_zero_distances(self):
        config=ExperimentConfig('classification','label','Fixture',('x',),('category',))
        data=Dataset(('x','category','label'),(('0','a','yes'),('1','b','no'),('2','a','yes'),('3','b','no')),'fixture','')
        report=distribution_shift(data,data,config)
        self.assertEqual(report['features'][0]['ks_distance'],0)
        self.assertEqual(report['features'][0]['psi'],0)
        self.assertEqual(report['features'][1]['total_variation'],0)
        self.assertEqual([r['feature'] for r in report['features']],['x','category'])

    def test_disjoint_ranges_and_unknown_categories_are_detected(self):
        config=ExperimentConfig('classification','label','Fixture',('x',),('category',))
        left=Dataset(('x','category','label'),tuple((str(i),'a','yes') for i in range(4)),'fixture','')
        right=Dataset(left.columns,tuple((str(i+10),'b','no') for i in range(4)),'fixture','')
        report=distribution_shift(left,right,config)
        self.assertEqual(report['features'][0]['ks_distance'],1)
        self.assertEqual(report['features'][0]['outside_reference_range_rate'],1)
        self.assertEqual(report['features'][1]['total_variation'],1)
        self.assertEqual(report['features'][1]['unseen_category_rate'],1)

    def test_large_constant_values_do_not_overflow_mean(self):
        report=numeric_distance([1e308,1e308],[1e308,1e308],10)
        self.assertEqual(report['reference_mean'],1e308)
        self.assertEqual(report['ks_distance'],0)
        json.dumps(report,allow_nan=False)
