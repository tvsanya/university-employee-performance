import unittest
import numpy as np
from university_analytics.data import Dataset,DatasetError
from university_analytics.experiment import ExperimentConfig
from university_analytics.uncertainty import grouped_bootstrap


class PerfectModel:
    classes=['a','b']
    def predict(self,data):return np.asarray([row[1] for row in data.rows])


class UncertaintyTests(unittest.TestCase):
    def test_groups_are_resampled_as_complete_units(self):
        data=Dataset(('id','label','x'),tuple((str(i//2),'a' if i%2==0 else 'b',str(i)) for i in range(20)),'fixture','')
        config=ExperimentConfig('classification','label','Test',('x',),group_column='id')
        report=grouped_bootstrap(data,config,PerfectModel(),repeats=50)
        self.assertEqual(report,grouped_bootstrap(data,config,PerfectModel(),repeats=50))
        self.assertEqual(report['groups'],10)
        self.assertEqual(report['resampling_unit'],'group')
        self.assertEqual(report['samples_missing_classes'],0)
        self.assertEqual(report['metrics']['macro_f1']['lower'],1)
        self.assertTrue(all(row['rows']==20 for row in report['samples']))

    def test_single_group_is_not_presented_as_independent_rows(self):
        data=Dataset(('id','label','x'),(('same','a','1'),('same','b','2')),'fixture','')
        config=ExperimentConfig('classification','label','Test',('x',),group_column='id')
        with self.assertRaises(DatasetError):grouped_bootstrap(data,config,PerfectModel(),repeats=50)
