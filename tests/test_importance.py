import unittest
import numpy as np
from university_analytics.data import Dataset, DatasetError
from university_analytics.experiment import ExperimentConfig
from university_analytics.importance import permutation_importance


class SignalModel:
    classes=['a','b']
    def predict(self,dataset):
        index=dataset.columns.index('signal')
        return np.asarray(['a' if row[index]=='0' else 'b' for row in dataset.rows])


class ImportanceTests(unittest.TestCase):
    def fixture(self):
        data=Dataset(('signal','noise','label'),tuple((str(i%2),str(i%3),'a' if i%2==0 else 'b') for i in range(100)),'fixture','')
        config=ExperimentConfig('classification','label','Test signal',('signal','noise'))
        return data,config

    def test_signal_is_detected_noise_is_zero_and_input_is_unchanged(self):
        data,config=self.fixture()
        report=permutation_importance(data,config,SignalModel(),5,42)
        self.assertEqual(report,permutation_importance(data,config,SignalModel(),5,42))
        self.assertEqual(report['features'][0]['feature'],'signal')
        self.assertGreater(report['features'][0]['macro_f1_drop_mean'],.3)
        self.assertEqual(report['features'][1]['macro_f1_drop_mean'],0)
        self.assertEqual(data.rows[0],('0','0','a'))

    def test_resource_and_seed_limits(self):
        data,config=self.fixture()
        for repeats,seed in [(0,42),(21,42),(5,-1),(True,42)]:
            with self.assertRaises(DatasetError):permutation_importance(data,config,SignalModel(),repeats,seed)
