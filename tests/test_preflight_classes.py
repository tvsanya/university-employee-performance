import csv
import json
import tempfile
import unittest
from pathlib import Path
from university_analytics.data import DatasetError, load_dataset
from university_analytics.jobs import TrainingManager
from university_analytics.simulation import generate_demo
from university_analytics.splitting import split_dataset


class PreflightClassTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        generate_demo(self.root/'data',employees=500,periods=2)
        self.source=self.root/'data/synthetic_employees.csv'
        self.config=json.loads((self.root/'data/experiment.json').read_text())
        self.original=self.source.read_bytes()
        self.manager=TrainingManager(self.root);self.addCleanup(self.manager.close)

    def relabel_partition(self, name, replace_with='rare'):
        data=load_dataset(str(self.source));group=self.config['group_column'];target=self.config['target']
        split=split_dataset(data,seed=self.config['seed'],group_column=group)
        group_i=data.columns.index(group);target_i=data.columns.index(target)
        groups={row[group_i] for row in split.partitions[name]}
        rows=[list(row) for row in data.rows]
        for row in rows:
            if row[group_i] in groups:row[target_i]=replace_with
        with self.source.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(data.columns);writer.writerows(rows)

    def test_unseen_validation_and_test_classes_fail_before_job_creation(self):
        for partition in ['validation','test']:
            with self.subTest(partition=partition):
                self.source.write_bytes(self.original)
                self.relabel_partition(partition)
                with self.assertRaisesRegex(DatasetError,'отсутствующие в train'):
                    self.manager.submit(self.source,self.config,'owner',2)
                self.assertEqual(list(self.manager.root.iterdir()),[])
                self.assertIsNone(self.manager.active)

    def test_counts_cover_rows_and_missing_validation_classes_are_explicit(self):
        self.relabel_partition('validation','высокая')
        data,config,summary=self.manager.prepare(self.source,self.config,2)
        self.assertEqual(summary['schema_version'],2)
        self.assertEqual(len(summary['class_labels']),3)
        for name,part in summary['partitions'].items():
            self.assertEqual(sum(summary['class_counts'][name].values()),part['rows'])
        self.assertTrue(any('validation:' in message and 'классов train' in message for message in summary['warnings']))
        self.assertEqual(list(self.manager.root.iterdir()),[])
