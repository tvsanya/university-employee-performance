import unittest
import numpy as np
from university_analytics.data import Dataset, DatasetError
from university_analytics.experiment import ExperimentConfig
from university_analytics.robustness import robustness_analysis


class StressModel:
    classes=['a','b']
    scale=np.asarray([1.0])
    def predict_proba(self,data):
        result=[]
        for row in data.rows:
            value=float(row[0]) if row[0] else 0
            result.append([.9,.1] if value < .5 else [.1,.9])
        return np.asarray(result)


class RobustnessTests(unittest.TestCase):
    def test_missing_signal_changes_prediction_without_modifying_target(self):
        data=Dataset(('signal','label'),tuple((str(i%2),'a' if i%2==0 else 'b') for i in range(100)),'fixture','')
        config=ExperimentConfig('classification','label','Signal',('signal',))
        report=robustness_analysis(data,config,StressModel())
        self.assertEqual(report,robustness_analysis(data,config,StressModel()))
        dropped=next(row for row in report['scenarios'] if row['scenario']=='complete_feature_missing')
        self.assertEqual(dropped['prediction_flip_rate'],.5)
        self.assertGreater(dropped['macro_f1_drop'],.5)
        self.assertEqual(data.rows[1],('1','b'))
        self.assertEqual(len(report['scenarios']),19)
        with self.assertRaises(DatasetError):robustness_analysis(data,config,StressModel(),repeats=0)
